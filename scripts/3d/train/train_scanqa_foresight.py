#!/usr/bin/env python3
"""Preview by default. Only --execute launches training, on user-selected GPUs."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import yaml

REPO = Path(__file__).resolve().parents[3]


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()] if path.suffix == ".jsonl" else json.load(stream)


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight(config, experiment):
    """CPU/file checks only. No torch import, GPU use, cache extraction or training."""
    data, model = config["data"], config["model"]
    root = Path(data["root"])
    originals = read_rows(root/data["train_original"])
    validation = read_rows(root/data["validation"])
    pairs = read_rows(root/data["train_paired"])
    if not originals or not validation or len(pairs) != len(originals):
        raise ValueError("Empty or differently sized pilot datasets")
    if [dict((k,v) for k,v in row.items() if k != "foresight") for row in pairs] != originals:
        raise ValueError("Paired rows must preserve the exact original QA rows and ordering")
    train_scenes, val_scenes = ({row["video"] for row in rows} for rows in (originals, validation))
    if train_scenes & val_scenes:
        raise ValueError("Training and validation scenes overlap")
    if any(row["metadata"]["dataset"].lower() != "scanqa" or len(row["conversations"]) != 2 for row in originals):
        raise ValueError("This configuration supports single-turn ScanQA only")
    cache = lambda scene: REPO/"data/scannet"/data["feature_id"]/scene.split("/")[-1]/"wan22_ti2v.pt"
    missing_train = [str(cache(s)) for s in sorted(train_scenes) if not cache(s).is_file()]
    if missing_train:
        raise FileNotFoundError("Missing train offline features: " + str(missing_train[:3]))
    for name in [model["initial_checkpoint"], model["vision_checkpoint"], model["wan_checkpoint"]]:
        if not (REPO/name/"config.json").is_file():
            raise FileNotFoundError(name)
    language = json.loads((REPO/model["initial_checkpoint"]/"config.json").read_text())
    wan = json.loads((REPO/model["wan_checkpoint"]/"config.json").read_text())
    if language.get("model_type") not in {"qwen3_5_text","llava_qwen35"} or language.get("hidden_size") != model["language_hidden_size"]:
        raise ValueError("Initial Qwen language architecture does not match the recorded configuration")
    if any(wan.get(key) != value for key,value in {"model_type":"ti2v","dim":3072,"in_dim":48,"out_dim":48,"num_layers":30}.items()):
        raise ValueError("Expected Wan2.2-TI2V-5B")
    if model["wan_context_dim"] != wan.get("text_dim",4096) or not (REPO/model["wan_checkpoint"]/"Wan2.2_VAE.pth").is_file():
        raise ValueError("Wan context dimension or VAE asset is incorrect")
    h, w, f = model["teacher_height"], model["teacher_width"], model["teacher_frames"]
    if h % 32 or w % 32 or f < 5 or (f-1) % 4 or not 1 <= model["queries"] <= 512:
        raise ValueError("Invalid teacher resolution/frame count or query count")
    # Inspect each distinct clip once. Full pair/actual-sampling checks run again in the loader.
    clips = {}
    for row in pairs:
        pair = row["foresight"]
        observed = [os.path.normpath(x) for x in pair["observed_frames"]]
        clip = pair["teacher_clip_frames"]
        if (pair["video"] != row["video"] or len(observed) != data["frame_count"]
                or len(clip) != f or clip[0] not in observed
                or set(clip[1:]) & set(observed) or clip[1:] != pair["heldout_target_frames"]
                or pair["target_mode"] != "heldout_view_clip_prediction_not_strict_future"):
            raise ValueError("Invalid or leaking teacher pair")
        if row["video"] in clips and clips[row["video"]] != clip:
            raise ValueError("Pilot expects one recorded teacher clip per scene")
        clips[row["video"]] = clip
    from PIL import Image
    for clip in clips.values():
        for name in clip:
            with Image.open(REPO/name) as image:
                image.verify()
    train_path = root/data["train_paired" if experiment["paired_data"] else "train_original"]
    return {"train_rows": len(originals), "train_scenes": len(train_scenes),
            "validation_rows": len(validation), "validation_scenes": len(val_scenes),
            "distinct_teacher_clips": len(clips), "train_path": str(train_path),
            "architecture": {"language_hidden":language["hidden_size"],"language_layers":language["num_hidden_layers"],
                             "wan_hidden":wan["dim"],"wan_layers":wan["num_layers"],"wan_heads":wan["num_heads"]},
            "validation_missing_feature_scenes": [s for s in sorted(val_scenes) if not cache(s).is_file()],
            "data_hashes": {name: file_digest(root/data[name]) for name in ["train_original", "train_paired", "validation"]},
            "protocol": "heldout_view_clip_prediction_not_strict_future"}


def training_arguments(config, experiment, data_yaml, output, run_name, gpu_count, max_steps=None):
    m, t, d = config["model"], config["training"], config["data"]
    if t["global_batch"] % (gpu_count*t["micro_batch"]):
        raise ValueError("Global batch must divide by GPU count * micro batch")
    if t["precision"] != "bf16" or t["torch_compile"]:
        raise ValueError("This pilot entry currently requires bf16 and torch_compile=false")
    args = dict(model_name_or_path=m["initial_checkpoint"], version="qwen35", data_path=str(data_yaml),
        vision_tower=m["vision_checkpoint"], image_folder="data", video_folder="data", embodiedscan_folder="data/embodiedscan/",
        mm_tunable_parts=t["tune"], mm_vision_tower_lr=t["vision_learning_rate"], mm_projector_type="mlp2x_gelu",
        mm_vision_select_layer=-2, mm_use_im_start_end=False, mm_use_im_patch_token=False,
        image_aspect_ratio="anyres_max_9", image_grid_pinpoints="(1x1),...,(6x6)", mm_patch_merge_type="spatial_unpad",
        bf16=True, tf32=True, output_dir=str(output), run_name=run_name, num_train_epochs=t["epochs"],
        max_steps=t["max_steps"] if max_steps is None else max_steps, report_to="none",
        per_device_train_batch_size=t["micro_batch"], per_device_eval_batch_size=1,
        gradient_accumulation_steps=t["global_batch"]//(gpu_count*t["micro_batch"]), eval_strategy="no",
        save_strategy="steps", save_steps=t["save_steps"], save_total_limit=2,
        learning_rate=t["learning_rate"], foresight_lr=t["foresight_learning_rate"], weight_decay=t["weight_decay"],
        optim=t["optimizer"], adam_beta1=t["adam_beta1"], adam_beta2=t["adam_beta2"],
        adam_epsilon=t["adam_epsilon"], max_grad_norm=t["max_grad_norm"],
        warmup_ratio=t["warmup_ratio"], lr_scheduler_type=t["scheduler"], logging_steps=1,
        model_max_length=t["model_max_length"], gradient_checkpointing=t["gradient_checkpointing"],
        dataloader_num_workers=t["data_workers"], lazy_preprocess=True, torch_compile=False,
        dataloader_drop_last=False, mm_newline_position="grid", add_spatial_instruction=True,
        force_sample=True, mm_spatial_pool_stride=2, world_position_embedding_type="avg-discrete-sin3d",
        object_feature_type="patch14-pe", group_by_task_length=False, frame_sampling_strategy="uniform",
        frames_upbound=d["frame_count"], use_generative_feature=True, generative_feat_dim=m["generative_feature_dim"],
        generative_projector_type="mlp2x_gelu", generative_feature_source="offline", generative_model_id=d["feature_id"],
        generative_cache_max_items=1, generative_profile=False, use_feat_fusion=True, feature_fusion_method="token_gated",
        seed=config["seed"], data_seed=config["seed"], foresight_enabled=experiment["queries_enabled"],
        foresight_num_queries=m["queries"], foresight_context_dim=m["wan_context_dim"],
        foresight_video_loss_weight=experiment["video_loss_weight"], foresight_wan_checkpoint=m["wan_checkpoint"],
        foresight_video_frames=m["teacher_frames"], foresight_video_height=m["teacher_height"],
        foresight_video_width=m["teacher_width"], foresight_flow_shift=m["flow_shift"],
        foresight_wan_gradient_checkpointing=m["teacher_gradient_checkpointing"])
    if t["deepspeed"]:
        args["deepspeed"] = t["deepspeed"]
    return [value for key, item in args.items() for value in ("--"+key, str(item))]


def check_loader(config, experiment, audit):
    """Read one real training row into the actual collator, with GPUs hidden."""
    import tempfile
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0,str(REPO))
    import torch
    from transformers import AutoTokenizer
    from llava import conversation as conversation_lib
    from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
    from llava.train.train_3d import DataArguments, LazySupervisedDataset, DataCollatorForSupervisedDataset
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["initial_checkpoint"],use_fast=True,local_files_only=True)
    tokenizer.model_max_length = config["training"]["model_max_length"]
    conversation_lib.default_conversation = conversation_lib.conv_templates["qwen35"].copy()
    with tempfile.TemporaryDirectory(prefix="foresight-loader-") as directory:
        path = Path(directory)/"train.yaml"
        path.write_text(yaml.safe_dump({"datasets":[{"json_path":audit["train_path"],"sampling_strategy":"all"}]}))
        data = DataArguments(data_path=str(path),image_folder="data",video_folder="data",embodiedscan_folder="data/embodiedscan/",
            frames_upbound=config["data"]["frame_count"],force_sample=True,is_multimodal=True,
            add_spatial_instruction=True,generative_model_id=config["data"]["feature_id"],frame_sampling_strategy="uniform")
        data.image_processor = SigLipImageProcessor()
        data.mm_use_im_start_end = False
        data.generative_feature_source = "offline"
        data.foresight_enabled = experiment["queries_enabled"]
        data.foresight_video_loss_weight = experiment["video_loss_weight"]
        data.foresight_video_frames = config["model"]["teacher_frames"]
        data.foresight_video_height = config["model"]["teacher_height"]
        data.foresight_video_width = config["model"]["teacher_width"]
        dataset = LazySupervisedDataset(str(path),tokenizer,data)
        before = json.dumps(dataset.list_data_dict[0],sort_keys=True)
        item = dataset._get_item(0)
        if json.dumps(dataset.list_data_dict[0],sort_keys=True) != before:
            raise RuntimeError("Dataset mutated the stored source annotation")
        batch = DataCollatorForSupervisedDataset(tokenizer)([item])
        output = {"row_id":item["id"], "batch_keys":sorted(batch),
            "text_tokens_before_visual_expansion":int(batch["input_ids"].shape[1]),
            "answer_tokens":int((batch["labels"]!=-100).sum()),
            "image_shapes":[list(x.shape) for x in batch["images"]],
            "video_tensor_shapes":{key:list(value.shape) for key,value in batch["video_dict"].items() if torch.is_tensor(value)},
            "teacher_shape":list(batch["foresight_pixels"].shape) if "foresight_pixels" in batch else None,
            "cuda_initialized":torch.cuda.is_initialized(),"optimizer_steps":0}
        if output["cuda_initialized"]:
            raise RuntimeError("CPU loader check unexpectedly initialized CUDA")
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/foresight/scanqa_pilot.yaml")
    parser.add_argument("--experiment", choices=["E00", "E01", "E03"], default="E03")
    parser.add_argument("--gpus", help="Required only with --execute, e.g. 0,1")
    parser.add_argument("--run-name")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-data", action="store_true", help="Read-only data and asset preflight")
    parser.add_argument("--check-loader", action="store_true", help="Read one actual QA/teacher batch on CPU; never train")
    parser.add_argument("--execute", action="store_true", help="Explicitly start training")
    cli = parser.parse_args()
    os.chdir(REPO)
    config_path = Path(cli.config)
    config = yaml.safe_load(config_path.read_text())
    if config["schema_version"] != 1:
        raise ValueError("Unsupported configuration schema")
    experiment = config["experiments"][cli.experiment]
    if cli.gpus and not re.fullmatch(r"[0-9]+(,[0-9]+)*", cli.gpus):
        parser.error("--gpus must be comma-separated integer IDs")
    gpu_ids = cli.gpus.split(",") if cli.gpus else ["UNSELECTED"]
    if len(set(gpu_ids)) != len(gpu_ids):
        parser.error("GPU IDs must be distinct")
    if cli.execute and not cli.gpus:
        parser.error("Choose GPUs explicitly before --execute")
    if cli.resume and not cli.run_name:
        parser.error("--resume requires --run-name")
    if cli.max_steps is not None and cli.max_steps != -1 and cli.max_steps < 1:
        parser.error("--max-steps must be -1 or positive")
    run_name = cli.run_name or f"qwen35-foresight-{cli.experiment.lower()}-{datetime.now():%Y%m%d-%H%M%S}"
    if not re.fullmatch(r"qwen35-[A-Za-z0-9_-]+", run_name):
        parser.error("Run name must start with qwen35- and use letters/digits/_/-")
    output = Path(config["training"]["output_root"])/run_name
    record = output/"foresight_record"
    data_yaml = record/"train.yaml"
    arguments = training_arguments(config, experiment, data_yaml, output, run_name, len(gpu_ids), cli.max_steps)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={len(gpu_ids)}",
               "wan_offline_tools/configure_wan_offline.py", "--train-entry", *arguments]
    audit = preflight(config, experiment) if cli.check_data or cli.check_loader or cli.execute else None
    if cli.check_loader:
        audit["loader"] = check_loader(config,experiment,audit)
    preview = {"mode": "EXECUTE" if cli.execute else "PREVIEW_ONLY", "experiment": cli.experiment,
               "config": config, "gpu_ids": gpu_ids, "output": str(output), "command": shlex.join(command), "preflight": audit}
    print(json.dumps(preview, ensure_ascii=False, indent=2), flush=True)
    if not cli.execute:
        return
    if cli.resume:
        if not list(output.glob("checkpoint-*")):
            raise FileNotFoundError("No periodic checkpoint to resume")
        previous = json.loads((record/"configuration.json").read_text())
        if previous["config"] != config or previous["experiment"] != cli.experiment:
            raise ValueError("Resume configuration differs from the original run")
        if previous["preflight"]["data_hashes"] != audit["data_hashes"]:
            raise ValueError("Pilot data changed since the original run")
        if len(previous["gpu_ids"]) != len(gpu_ids):
            raise ValueError("Resume requires the same GPU process count for the ZeRO checkpoint")
    elif output.exists():
        raise FileExistsError("Choose a new run name or explicitly resume")
    record.mkdir(parents=True, exist_ok=cli.resume)
    train_path = audit["train_path"]
    data_yaml.write_text(yaml.safe_dump({"datasets": [{"json_path": train_path, "sampling_strategy": "all"}]}))
    def git(*args):
        return subprocess.run(["git", *args], text=True, capture_output=True, check=True).stdout.strip()
    packages = {}
    for name in ["torch", "transformers", "diffusers", "accelerate", "deepspeed", "safetensors"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not_installed"
    # Record tracked source and newly added source; no data, keys or environment secrets.
    paths = sorted(set(git("ls-files").splitlines() + git("ls-files", "--others", "--exclude-standard").splitlines()))
    hashes = {p: file_digest(Path(p)) for p in paths if Path(p).is_file() and Path(p).suffix in {".py", ".sh", ".yaml", ".json"}}
    record_data = {**preview, "recorded_at": datetime.now(timezone.utc).isoformat(), "git_commit": git("rev-parse", "HEAD"),
                   "git_status": git("status", "--short"), "source_sha256": hashes, "packages": packages,
                   "config_sha256": file_digest(config_path), "argv": arguments}
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    destination = record/(f"resume_{stamp}.json" if cli.resume else "configuration.json")
    if cli.resume and previous["source_sha256"] != hashes:
        raise ValueError("Source changed since the saved run; start a new experiment instead")
    destination.write_text(json.dumps(record_data, ensure_ascii=False, indent=2))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": cli.gpus, "TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1",
           "TOKENIZERS_PARALLELISM": "false", "QWEN35_FULL_FINAL_CHECKPOINT": "1", "QWEN35_SUPERVISED_LOGITS_ONLY": "1",
           "OMP_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(REPO)+os.pathsep+os.environ.get("PYTHONPATH", ""),
           "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")}
    # No background worker, retry loop or automatic follow-up job.
    with (record/f"train_{stamp}.log").open("a") as log:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            status = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            status = process.wait()
        (record/f"exit_{stamp}.json").write_text(json.dumps({"exit_code": status, "training_started": True}))
    raise SystemExit(status)


if __name__ == "__main__":
    main()
