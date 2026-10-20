#!/usr/bin/env python3
"""Native Qwen3.5 ScanQA generation, without Ray or a video-loss teacher.

Preview by default; --execute and --gpu are required to load a GPU model.
The answer JSONL preserves the existing ScanQA metric input schema.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]


def generation_prompt(row, tokenizer, add_spatial=True):
    from llava.constants import DEFAULT_IMAGE_TOKEN
    question = row["conversations"][0]["value"]
    if add_spatial:
        spatial = ("The video captures 3D spatial information of a scene. Please focus on the spatial "
                   "relationships in the video and answer the following questions.")
        # Match train_3d.py exactly, including the source question's whitespace.
        question = DEFAULT_IMAGE_TOKEN+"\n"+spatial+"\n"+question.replace(DEFAULT_IMAGE_TOKEN, "")
    messages = [{"role":"system", "content":"You are a helpful assistant."},
                {"role":"user", "content":question}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Saved full VEGA checkpoint")
    parser.add_argument("--question-file", default="/data/zhdu/vega3d_repro/experiments/scanqa_foresight_pilot_20261020_v1/data/val_original.json")
    parser.add_argument("--answer-file", required=True)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    os.chdir(REPO)
    sys.path.insert(0,str(REPO))
    question_file = Path(args.question_file)
    rows = json.loads(question_file.read_text())
    stored = json.loads((Path(args.model_path)/"config.json").read_text())
    if stored.get("model_type") != "llava_qwen35" or not stored.get("mm_vision_tower"):
        raise ValueError("Use a saved multimodal VEGA-Qwen3.5 checkpoint, not the raw language-only weights")
    if not rows or args.max_new_tokens < 1:
        raise ValueError("Empty question set or invalid generation length")
    if any(row["metadata"]["dataset"].lower() != "scanqa" for row in rows):
        raise ValueError("Only ScanQA questions are supported")
    feature_id = stored["generative_model_id"]
    if stored.get("generative_feature_source") != "offline":
        raise ValueError("This matched-pilot evaluator requires the saved offline feature protocol")
    missing = [scene for scene in sorted({row["video"] for row in rows})
        if not (REPO/"data/scannet"/feature_id/scene.split("/")[-1]/"wan22_ti2v.pt").is_file()]
    preview = {"mode":"EXECUTE" if args.execute else "PREVIEW_ONLY", "model":args.model_path,
        "question_file":str(question_file), "question_sha256":hashlib.sha256(question_file.read_bytes()).hexdigest(),
        "rows":len(rows), "gpu":args.gpu, "frames":stored.get("frames_upbound",32),
        "foresight_queries":stored.get("foresight_num_queries",0) if stored.get("foresight_enabled") else 0,
        "teacher_loaded":False, "missing_feature_scenes":missing,
        "decoding":{"do_sample":False,"num_beams":1,"max_new_tokens":args.max_new_tokens},
        "answer_file":args.answer_file, "recorded_at":datetime.now(timezone.utc).isoformat()}
    print(json.dumps(preview, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return
    if args.gpu is None or args.gpu < 0:
        parser.error("Choose a non-negative --gpu explicitly")
    if missing:
        raise FileNotFoundError("Extract validation offline features before evaluation: "+str(missing))
    output = Path(args.answer_file)
    meta = output.with_suffix(output.suffix+".meta.json")
    if output.exists() or meta.exists():
        raise FileExistsError("Choose a fresh answer file; existing results are never overwritten")
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), TRANSFORMERS_OFFLINE="1", HF_HUB_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false")
    import torch
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.video_utils import VideoProcessor, merge_video_dict
    from llava.eval.generative_eval_utils import move_video_dict_to_device
    tokenizer, model, processor, _ = load_pretrained_model(args.model_path, None, Path(args.model_path).name,
        torch_dtype="bfloat16", device_map={"":"cuda:0"}, attn_implementation="sdpa")
    model.eval()
    video_processor = VideoProcessor(video_folder="data", annotation_dir="data/embodiedscan/",
        voxel_size=getattr(model.config,"voxel_size",0.1),
        min_xyz_range=getattr(model.config,"min_xyz_range",[-15,-15,-5]),
        max_xyz_range=getattr(model.config,"max_xyz_range",[15,15,5]), frame_sampling_strategy="uniform")
    output.parent.mkdir(parents=True,exist_ok=True)
    meta.write_text(json.dumps(preview, ensure_ascii=False, indent=2))
    with output.open("x") as stream, torch.inference_mode():
        for index, row in enumerate(rows):
            prompt = generation_prompt(row, tokenizer, getattr(model.config,"add_spatial_instruction",True))
            ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
            video = video_processor.process_3d_video(row["video"], processor, force_sample=True,
                frames_upbound=preview["frames"], generative_model_id=feature_id, generative_feature_source="offline")
            images, video = move_video_dict_to_device(merge_video_dict([video]), model.device, torch.bfloat16)
            generated = model.generate(inputs=ids, images=images, modalities=["video"], video_dict=video,
                do_sample=False, num_beams=1, max_new_tokens=args.max_new_tokens, use_cache=True)
            answer = tokenizer.batch_decode(generated,skip_special_tokens=True)[0].strip()
            if model.__dict__["_foresight_teacher_runtime"] is not None:
                raise RuntimeError("Evaluation unexpectedly loaded the video-loss teacher")
            item = {"dataset":row["metadata"]["dataset"], "sample_id":row["id"],
                "prompt":row["conversations"][0]["value"], "pred_response":answer,
                "gt_response":row["conversations"][1]["value"], "model_id":Path(args.model_path).name,
                "question_type":row["metadata"].get("question_type","unknown")}
            stream.write(json.dumps(item,ensure_ascii=False)+"\n")
            stream.flush()
            print(f"{index+1}/{len(rows)} {row['id']}: {answer}",flush=True)


if __name__ == "__main__":
    main()
