#!/usr/bin/env python3
"""Patch only offline feature routing/packing and create a dedicated smoke launcher.

Does not replace the user's Qwen3.5 model or trainer files with upstream files.
All candidates are validated before any source file is written; source backups
are retained beside each edited file.
"""
import ast
from datetime import datetime
import json
import os
import sys
import inspect
from pathlib import Path
import shutil
import subprocess
import textwrap

LOADER_MARKER = "# VEGA-WAN-OFFLINE: validated unprojected T2V feature payload."
LOADER = '''        # VEGA-WAN-OFFLINE: validated unprojected T2V feature payload.
        if str(model_id).startswith("wan22_ti2v_offline_uniform"):
            if not re.fullmatch(r"wan22_ti2v_offline_uniform[0-9]+_[0-9]+x[0-9]+_t[0-9]+_b[0-9]+_s[0-9]+_v1", str(model_id)):
                raise ValueError(f"Invalid Wan offline feature ID: {model_id}")
            scene = video_id.split("/")[-1]
            feature_path = os.path.join("data/scannet", model_id, scene, "wan22_ti2v.pt")
            payload = torch.load(feature_path, map_location="cpu", weights_only=True)
            metadata = payload["metadata"]
            if metadata["video_id"] != video_id:
                raise ValueError(f"Offline feature scene mismatch: {feature_path}")
            if metadata["plan"]["feature_id"] != model_id:
                raise ValueError(f"Offline feature configuration mismatch: {feature_path}")
            if frame_files is None:
                raise ValueError("Frame paths are required to validate Wan offline alignment")
            expected_frames = [os.path.normpath(p) for p in frame_files]
            if metadata["frame_files"] != expected_frames:
                raise ValueError(f"Offline feature frame order/sampling mismatch: {feature_path}")
            feature = payload["features"]
            if not torch.is_tensor(feature) or tuple(feature.shape) != (len(frame_files), 3072, 14, 14):
                raise ValueError(f"Invalid Wan offline feature shape: {feature_path}")
            if not torch.isfinite(feature).all():
                raise ValueError(f"Non-finite Wan offline feature: {feature_path}")
            return feature.to(torch.bfloat16)

'''

OFFLINE_MARKER = "# VEGA-WAN-OFFLINE: preserve video alignment and projector dtype."
OFFLINE = '''                if generative_feature_source == "offline":
                    # VEGA-WAN-OFFLINE: preserve video alignment and projector dtype.
                    if len(video_idx_in_batch) > 0:
                        if video_dict is None or video_dict.get("generative_feature") is None:
                            raise ValueError("Offline generative features are required; extract them before training")
                        generative_features = video_dict["generative_feature"]
                        if generative_features.ndim != 5:
                            raise ValueError(f"Expected [B,V,C,H,W] offline features, got {generative_features.shape}")
                        B, V, Cg, Hg, Wg = generative_features.shape
                        expected_dim = int(getattr(self.config, "generative_feat_dim", 3072))
                        if B != len(video_idx_in_batch) or Cg != expected_dim:
                            raise ValueError(f"Offline feature batch/channel mismatch: {generative_features.shape}")
                        for batch_i in video_idx_in_batch:
                            if images_list[batch_i].shape[0] != V:
                                raise ValueError("Offline features and sampled video frames have different lengths")
                        projector = self.get_model().generative_projector
                        generative_features = generative_features.to(
                            device=concat_images.device,
                            dtype=next(projector.parameters()).dtype,
                        )
                        generative_features = generative_features.reshape(B * V, Cg, Hg * Wg)
                        generative_features = generative_features.permute(0, 2, 1).contiguous()
                        generative_features = projector(generative_features)
                        offline_splits = torch.split(generative_features, [V] * B)
                        for batch_i, features in zip(video_idx_in_batch, offline_splits):
                            generative_features_list[batch_i] = features
'''


def replace_once(source, old, new, label):
    if source.count(old) != 1:
        raise ValueError(f"Expected one {label}; found {source.count(old)}. No pending edits have been written.")
    return source.replace(old, new, 1)


def patch_video(source):
    if LOADER_MARKER in source:
        return source
    source = replace_once(source,
        "    def get_generative_features(self, video_id, model_id='seva'):\n",
        "    def get_generative_features(self, video_id, model_id='seva', frame_files=None):\n" + LOADER,
        "offline loader method")
    source = replace_once(source,
        "self.get_generative_features(video_id, model_id=generative_model_id)",
        "self.get_generative_features(video_id, model_id=generative_model_id, frame_files=frame_files)",
        "offline loader call")
    # Features are small per scene but an unlimited all-scene RAM cache can grow
    # substantially. Bound this new Wan cache without affecting legacy encoders.
    anchor = "                self.generative_feature[video_id] = generative_feature\n"
    extra = '''                self.generative_feature[video_id] = generative_feature
                if str(generative_model_id).startswith("wan22_ti2v_offline_uniform"):
                    while len(self.generative_feature) > 4:
                        self.generative_feature.pop(next(iter(self.generative_feature)))
'''
    return replace_once(source, anchor, extra, "scene cache assignment")


def patch_arch(source):
    if OFFLINE_MARKER in source:
        return source
    # Repair the earlier misplaced online branch if that fix was not applied.
    old = '                    elif generative_feature_source == "online":\n                      if len(video_idx_in_batch) > 0:'
    new = '                elif generative_feature_source == "online":\n                    if len(video_idx_in_batch) > 0:'
    if old in source:
        source = replace_once(source, old, new, "misindented online branch")
    start_text = '                if generative_feature_source == "offline":\n'
    end_text = '                elif generative_feature_source == "online":\n'
    if source.count(start_text) != 1 or source.count(end_text) != 1:
        raise ValueError("Cannot identify the offline/online packing branches; source files remain unchanged")
    start = source.index(start_text)
    end = source.index(end_text, start)
    return source[:start] + OFFLINE + source[end:]


def patch_exception(source):
    old = '''        except Exception as e:
            import traceback; traceback.print_exc()
            exit()'''
    new = '''        except Exception:
            import traceback
            traceback.print_exc()
            raise'''
    if old in source:
        return replace_once(source, old, new, "training exception handler")
    # Accept an existing corrected handler; reject a silent exit in the trainer
    # training try/except rather than claiming the launcher will detect failure.
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        trains = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                     and isinstance(n.func.value, ast.Name) and n.func.value.id == "trainer"
                     and n.func.attr == "train" for b in node.body for n in ast.walk(b))
        if trains:
            for handler in node.handlers:
                if not any(isinstance(n, ast.Raise) for n in ast.walk(handler)):
                    raise ValueError("Training exceptions must be re-raised; inspect train_3d.py before using this launcher")
    return source


def configure(repo):
    os.chdir(repo)
    pointer = json.loads(Path("wan_offline_tools/generated/qwen35_wan_offline_manifest.json").read_text())
    plan = json.loads(Path(pointer["manifest"]).read_text())
    for scene in plan["scenes"]:
        path = Path("data/scannet") / plan["feature_id"] / scene.split("/")[-1] / "wan22_ti2v.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Run extraction first: {path}")
    candidates = []
    for name, patch in [("llava/video_utils.py", patch_video),
                        ("llava/model/llava_arch.py", patch_arch),
                        ("llava/train/train_3d.py", patch_exception)]:
        path = Path(name)
        source = path.read_text()
        updated = patch(source)
        compile(updated, str(path), "exec")
        if updated != source:
            candidates.append((path, updated))
    target = Path("scripts/3d/train/train_qwen35_wan22.sh")
    if not target.is_file():
        raise FileNotFoundError("Install the unified Qwen3.5 launcher first: " + str(target))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    for path, updated in candidates:
        backup = path.with_name(path.name + ".bak-offline-" + stamp)
        shutil.copy2(path, backup)
        path.write_text(updated)
        print("Patched:", path, "Backup:", backup)
    print("Unified Qwen3.5 launcher:", target)
    print("Run: bash", target)



def install_checkpoint_compat(trainer_class):
    """Adapt only the old optional metrics argument to the installed HF API."""
    native = trainer_class._save_checkpoint
    if "metrics" not in inspect.signature(native).parameters:
        def compatible_save(self, model, trial, metrics=None):
            if metrics is not None:
                raise ValueError("This compatibility entry supports original evaluation_strategy=no only")
            return native(self, model, trial)
        trainer_class._save_checkpoint = compatible_save
        print("[Qwen3.5] Checkpoint API compatibility active; native model/optimizer/RNG saving preserved.", flush=True)



def install_rng_checkpoint_compat(trainer_class):
    import torch
    import numpy as np
    import _codecs
    import functools
    native_rng = trainer_class._load_rng_state
    @functools.wraps(native_rng)
    def load_rng_with_numpy_types(self, checkpoint):
        # Scope the allowlist to the native RNG restore call. Keep weights_only
        # enabled and use only the NumPy/codec types in the local NumPy RNG state.
        rng_types = [np.core.multiarray._reconstruct, np.ndarray, np.dtype,
                     type(np.dtype(np.uint32)), _codecs.encode]
        previous = list(torch.serialization.get_safe_globals())
        torch.serialization.add_safe_globals(rng_types)
        try:
            return native_rng(self, checkpoint)
        finally:
            torch.serialization.clear_safe_globals()
            torch.serialization.add_safe_globals(previous)
    trainer_class._load_rng_state = load_rng_with_numpy_types
    print("[Qwen3.5] RNG checkpoint compatibility active: scoped NumPy RNG types; weights_only stays enabled.", flush=True)


def install_final_checkpoint(trainer_class):
    import functools
    native_train = trainer_class.train
    @functools.wraps(native_train)
    def train_with_final_checkpoint(self, *args, **kwargs):
        result = native_train(self, *args, **kwargs)
        if self.state.global_step > 0:
            # Every rank calls the native distributed save, including when the
            # last step already had a periodic checkpoint.
            self._save_checkpoint(self.model, None)
        return result
    trainer_class.train = train_with_final_checkpoint
    print("[Qwen3.5] Native full checkpoint will also be saved at successful training end.", flush=True)


if __name__ == "__main__":
    repo = Path(__file__).resolve().parent.parent
    if sys.argv[1:2] == ["--train-entry"]:
        import runpy
        os.chdir(repo)
        sys.path.insert(0, str(repo))
        if os.environ.get('QWEN35_SUPERVISED_LOGITS_ONLY', '0') == '1':
            from wan_offline_tools.qwen35_memory_compat import install_supervised_logits_only
            install_supervised_logits_only()
        from transformers import Trainer
        install_checkpoint_compat(Trainer)
        install_rng_checkpoint_compat(Trainer)
        if os.environ.get("QWEN35_FULL_FINAL_CHECKPOINT") == "1":
            install_final_checkpoint(Trainer)
        sys.argv = [str(repo / "llava/train/train_3d.py")] + sys.argv[2:]
        runpy.run_path(sys.argv[0], run_name="__main__")
    else:
        configure(repo)
