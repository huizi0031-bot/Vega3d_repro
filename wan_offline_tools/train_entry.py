#!/usr/bin/env python3
"""Qwen3.5 training entry with checkpoint and supervised-logit compatibility."""
import inspect
import os
from pathlib import Path
import sys


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
    native_save = trainer_class._save_checkpoint
    @functools.wraps(native_save)
    def save_and_track_step(self, *args, **kwargs):
        result = native_save(self, *args, **kwargs)
        self.__dict__["_qwen35_last_complete_checkpoint_step"] = self.state.global_step
        return result
    trainer_class._save_checkpoint = save_and_track_step
    @functools.wraps(native_train)
    def train_with_final_checkpoint(self, *args, **kwargs):
        result = native_train(self, *args, **kwargs)
        if (self.state.global_step > 0 and
                self.__dict__.get("_qwen35_last_complete_checkpoint_step") != self.state.global_step):
            # Every rank must participate, but a successful native final-step
            # save already contains the model, optimizer and RNG state.
            self._save_checkpoint(self.model, None)
        return result
    trainer_class.train = train_with_final_checkpoint
    print("[Qwen3.5] Native full checkpoint will also be saved at successful training end.", flush=True)



def main():
    import runpy
    repo = Path(__file__).resolve().parent.parent
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    if os.environ.get("QWEN35_SUPERVISED_LOGITS_ONLY") == "1":
        from wan_offline_tools.qwen35_memory_compat import install_supervised_logits_only
        install_supervised_logits_only()
    from transformers import Trainer
    install_checkpoint_compat(Trainer)
    install_rng_checkpoint_compat(Trainer)
    if os.environ.get("QWEN35_FULL_FINAL_CHECKPOINT") == "1":
        install_final_checkpoint(Trainer)
    sys.argv = [str(repo/"llava/train/train_3d.py")] + sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
