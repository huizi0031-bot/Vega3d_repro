#!/usr/bin/env python3
"""Shared scene-understanding training with optional foresight query/video supervision."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys

import yaml

REPO = Path(__file__).resolve().parents[3]


sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_utils import training_arguments
from data_checks import preflight, file_digest, check_weight_files, resolve_path


def resolve_config_paths(config):
    for key in ("train", "validation"):
        if config["data"].get(key):
            config["data"][key] = str(resolve_path(config["data"][key]))
    config["training"]["output_root"] = str(resolve_path(config["training"]["output_root"]))
    return config


def resolve_experiment(config, mode=None):
    mode = mode or config.get("foresight", {}).get("mode", "baseline")
    weight = float(config.get("foresight", {}).get("video_loss_weight", 0.1)) if mode == "video" else 0.0
    if mode not in {"baseline", "queries", "video"} or not 0 <= weight < float("inf") or (mode == "video" and weight == 0):
        raise ValueError("Use baseline, queries, or video with a positive finite video_loss_weight")
    head = config["model"].get("ground_head_type")
    if mode != "baseline" and head is not None:
        raise ValueError("Object-proposal grounding is not connected to foresight; use ground_head_type: null")
    return dict(queries_enabled=mode != "baseline", video_loss_weight=weight)


def check_loader(config, experiment, audit, pair_plan):
    """Read a real training row into the actual collator, with GPUs hidden."""
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
        path.write_text(yaml.safe_dump(audit["datasets"]))
        data = DataArguments(data_path=str(path),image_folder="data",video_folder="data",embodiedscan_folder="data/embodiedscan/",
            frames_upbound=config["data"]["frame_count"],force_sample=True,is_multimodal=True,
            add_spatial_instruction=True,generative_model_id=config["data"]["feature_id"],frame_sampling_strategy="uniform")
        data.foresight_max_translation = config["data"].get("max_translation", 0.5)
        data.foresight_max_rotation = config["data"].get("max_rotation", 30.0)
        if experiment["video_loss_weight"] > 0:
            pair_file = Path(directory) / "scene_pairs.json"
            pair_file.write_text(json.dumps(pair_plan, ensure_ascii=False, indent=2))
            data.foresight_pair_cache = str(pair_file)
        data.image_processor = SigLipImageProcessor()
        data.mm_use_im_start_end = False
        data.generative_feature_source = "offline"
        data.foresight_enabled = experiment["queries_enabled"]
        data.foresight_video_loss_weight = experiment["video_loss_weight"]
        data.foresight_video_frames = config["model"].get("teacher_frames", 5)
        data.foresight_video_height = config["model"].get("teacher_height", 256)
        data.foresight_video_width = config["model"].get("teacher_width", 320)
        import random
        random.seed(config["seed"])
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
    parser.add_argument("--config", default="configs/3d/foresight.yaml")
    parser.add_argument("--mode", choices=["baseline", "queries", "video"])
    parser.add_argument("--data-path", help="Override data.train: JSON, JSONL or datasets YAML")
    parser.add_argument("--validation-path", help="Override data.validation: JSON, JSONL or datasets YAML")
    parser.add_argument("--audit-file", help="Write the CPU data audit as JSON, including failures")
    parser.add_argument("--ground-head", choices=["none", "infonce", "mlp", "score"])
    parser.add_argument("--gpus", help="Required only with --execute, e.g. 0,1")
    parser.add_argument("--run-name")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--master-port", type=int, help="Optional local single-node rendezvous port")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-data", action="store_true", help="Read-only data and asset preflight")
    parser.add_argument("--check-loader", action="store_true", help="Read an actual input/teacher batch on CPU; never train")
    parser.add_argument("--execute", action="store_true", help="Explicitly start training")
    cli = parser.parse_args()
    os.chdir(REPO)
    config_path = Path(cli.config)
    config = yaml.safe_load(config_path.read_text())
    if config["schema_version"] != 1:
        raise ValueError("Unsupported configuration schema")
    config = resolve_config_paths(config)
    if cli.data_path:
        config["data"]["train"] = str(resolve_path(cli.data_path))
    if cli.validation_path:
        config["data"]["validation"] = str(resolve_path(cli.validation_path))
    if cli.audit_file and not (cli.check_data or cli.check_loader or cli.execute):
        parser.error("--audit-file requires --check-data, --check-loader or --execute")
    if cli.ground_head:
        config["model"]["ground_head_type"] = None if cli.ground_head == "none" else cli.ground_head
    mode = cli.mode or config.get("foresight", {}).get("mode", "baseline")
    experiment = resolve_experiment(config, mode)
    if cli.execute and (cli.check_data or cli.check_loader):
        parser.error("Run --check-data/--check-loader separately from --execute")
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
    if cli.master_port is not None and not 1 <= cli.master_port <= 65535:
        parser.error("--master-port must be between 1 and 65535")
    if cli.master_port is None:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            master_port = listener.getsockname()[1]
    else:
        master_port = cli.master_port
    run_name = cli.run_name or f"scene-{mode}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_name):
        parser.error("Run name must use letters/digits/_/- and start with a letter or digit")
    output = Path(config["training"]["output_root"])/run_name
    record = output/"foresight_record"
    data_yaml = record/"train.yaml"
    arguments = training_arguments(config, experiment, data_yaml, output, run_name, len(gpu_ids), cli.max_steps)
    # Numeric hostnames such as 8709 resolve as 0.0.34.5 in elastic rendezvous.
    # This launcher is single-node, so use an explicit static loopback store.
    command = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--node_rank=0",
               "--rdzv_backend=static", "--master_addr=127.0.0.1", f"--master_port={master_port}",
               f"--nproc_per_node={len(gpu_ids)}",
               "wan_offline_tools/train_entry.py", *arguments]
    # Preflight and loader checks run on CPU even when the child will train on GPUs.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    audit = preflight(config, experiment) if cli.check_data or cli.check_loader or cli.execute else None
    pair_plan = audit.pop("_pair_plan") if audit is not None else None
    if cli.audit_file:
        audit_path = resolve_path(cli.audit_file)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    if cli.check_loader and audit["ready_for_training"]:
        audit["loader"] = check_loader(config, experiment, audit, pair_plan)
        if cli.audit_file:
            audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    preview = {"mode": "EXECUTE" if cli.execute else "PREVIEW_ONLY", "mode_name": mode,
               "config": config, "gpu_ids": gpu_ids, "output": str(output), "command": shlex.join(command), "preflight": audit}
    print(json.dumps(preview, ensure_ascii=False, indent=2), flush=True)
    if audit is not None and not audit["ready_for_training"]:
        print("Training assets are incomplete; no samples were dropped and no training was started.", file=sys.stderr)
        raise SystemExit(1)
    if not cli.execute:
        return
    if cli.resume:
        if not list(output.glob("checkpoint-*")):
            raise FileNotFoundError("No periodic checkpoint to resume")
        previous = json.loads((record/"configuration.json").read_text())
        if previous["config"] != config or previous["mode_name"] != mode or previous["argv"] != arguments:
            raise ValueError("Resume configuration differs from the original run")
        if previous["preflight"]["data_hashes"] != audit["data_hashes"]:
            raise ValueError("Data changed since the original run")
        if len(previous["gpu_ids"]) != len(gpu_ids):
            raise ValueError("Resume requires the same GPU process count for the ZeRO checkpoint")
    elif output.exists():
        raise FileExistsError("Choose a new run name or explicitly resume")
    pair_file = record / "scene_pairs.json"
    if cli.resume and experiment["video_loss_weight"] > 0:
        if not pair_file.is_file() or json.loads(pair_file.read_text()) != pair_plan:
            raise ValueError("Saved foresight selections differ from the current dataset/assets")
    record.mkdir(parents=True, exist_ok=cli.resume)
    if experiment["video_loss_weight"] > 0 and not cli.resume:
        pair_file.write_text(json.dumps(pair_plan, ensure_ascii=False, indent=2))
    data_yaml.write_text(yaml.safe_dump(audit["datasets"]))
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
    attempt = len(list(record.glob("resume_*.json"))) + 1
    destination = record/(f"resume_{attempt}.json" if cli.resume else "configuration.json")
    if cli.resume and previous["source_sha256"] != hashes:
        raise ValueError("Source changed since the saved run; start a new experiment instead")
    destination.write_text(json.dumps(record_data, ensure_ascii=False, indent=2))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": cli.gpus, "TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1",
           "TOKENIZERS_PARALLELISM": "false", "QWEN35_FULL_FINAL_CHECKPOINT": "1", "QWEN35_SUPERVISED_LOGITS_ONLY": "1",
           "OMP_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(REPO)+os.pathsep+os.environ.get("PYTHONPATH", ""),
           "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")}
    with (record/"train.log").open("a") as log:
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
        (record/"exit.json").write_text(json.dumps({"exit_code": status, "training_started": True}))
    raise SystemExit(status)


if __name__ == "__main__":
    main()
