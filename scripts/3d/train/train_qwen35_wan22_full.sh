#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
REPO_ROOT="$(pwd)"
QWEN35_ENV_ROOT="${QWEN35_ENV_ROOT:-$(dirname "$(dirname "$REPO_ROOT")")/envs/vega3d-qwen35}"
export PATH="$QWEN35_ENV_ROOT/bin:$PATH"
STAGE=all
GPUS=2,3
RUN_NAME=""
OUTPUT_ROOT="$(dirname "$(dirname "$REPO_ROOT")")/checkpoints"
RESUME=0
DRY_RUN=0
DATA_YAML="${QWEN35_FULL_DATA_PATH:-scripts/3d/train/multi.yaml}"
MAX_STEPS="${QWEN35_FULL_MAX_STEPS:--1}"
NUM_TRAIN_EPOCHS=1
GLOBAL_BATCH_SIZE="${QWEN35_FULL_GLOBAL_BATCH:-128}"
SAVE_STEPS="${QWEN35_FULL_SAVE_STEPS:-500}"
DATA_WORKERS="${QWEN35_FULL_DATA_WORKERS:-1}"
COMPILE_MODEL="${QWEN35_FULL_COMPILE:-True}"
COMPILE_ARGS=(--torch_compile "$COMPILE_MODEL")
case "${COMPILE_MODEL,,}" in
    true|1) COMPILE_ARGS+=(--torch_compile_backend inductor);;
    false|0) ;;
    *) echo 'QWEN35_FULL_COMPILE must be True or False' >&2; exit 2;;
esac
FRAMES=32
FEATURE_SOURCE=offline
FEATURE_ID=wan22_ti2v_offline_uniform32_1280x704_t300_b20_s42_v1
GENERATIVE_PROFILE="${GENERATIVE_PROFILE:-False}"
while (($#)); do
    case "$1" in
        --stage|--gpus|--run-name|--output-root)
            [[ $# -ge 2 ]] || exit 2
            case "$1" in
                --stage) STAGE="$2";; --gpus) GPUS="$2";;
                --run-name) RUN_NAME="$2";; --output-root) OUTPUT_ROOT="$2";;
            esac
            shift 2;;
        --resume) RESUME=1; shift;;
        --dry-run) DRY_RUN=1; shift;;
        --help|-h)
            echo 'Qwen3.5 full-data training: --stage all|plan|extract|train --gpus 2,3 --run-name NAME --output-root PATH --resume --dry-run'
            echo 'Original full hyperparameters: 32 frames, global batch 128, 1 epoch, max length 32768, checkpoint every 500 steps.'
            echo 'Offline Wan extraction and training use the original repository encoder, train_3d.py and sampler.'
            exit 0;;
        *) echo "Unknown option: $1" >&2; exit 2;;
    esac
done
[[ "$STAGE" =~ ^(all|plan|extract|train)$ && "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]] || exit 2
IFS=',' read -r -a GPU_IDS <<< "$GPUS"
NUM_GPUS="${#GPU_IDS[@]}"
for ((i=0;i<NUM_GPUS;i++)); do
    for ((j=i+1;j<NUM_GPUS;j++)); do [[ "${GPU_IDS[i]}" != "${GPU_IDS[j]}" ]] || exit 2; done
done
[[ "$GLOBAL_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || exit 2
((GLOBAL_BATCH_SIZE % NUM_GPUS == 0)) || { echo 'Global batch must divide by GPU count'; exit 2; }
[[ -n "$RUN_NAME" ]] || RUN_NAME="qwen35-wan22-full-train-gpu${GPUS//,/}-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)-$$"
[[ "$RUN_NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
[[ "$RUN_NAME" == qwen35-* ]] || RUN_NAME="qwen35-$RUN_NAME"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_NAME"
LOG_FILE="$OUTPUT_ROOT/$RUN_NAME.log"
RECORD_DIR="wan_full_smoke_tools/full_training_runs/$RUN_NAME"
GEN_ARGS=(--generative_feature_source offline --generative_model_id "$FEATURE_ID")
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
echo "[Qwen3.5] STAGE=$STAGE GPUS=$GPUS DATA=$DATA_YAML FRAMES=32 GLOBAL_BATCH=$GLOBAL_BATCH_SIZE EPOCHS=1 MAX_STEPS=$MAX_STEPS"
echo "[Qwen3.5] CHECKPOINT=$OUTPUT_DIR SAVE_STEPS=$SAVE_STEPS LOG=$LOG_FILE"
if ((DRY_RUN)); then echo "[Qwen3.5] ACCUMULATION=$((GLOBAL_BATCH_SIZE/NUM_GPUS)) MODEL_MAX_LENGTH=32768 COMPILE=$COMPILE_MODEL";exit 0;fi
mkdir -p "$OUTPUT_ROOT" "$RECORD_DIR"
export QWEN35_FULL_RECORD_DIR="$RECORD_DIR" QWEN35_FULL_OUTPUT_DIR="$OUTPUT_DIR" QWEN35_FULL_STAGE="$STAGE" QWEN35_FULL_RUN_NAME="$RUN_NAME" QWEN35_FULL_GPU_IDS="$GPUS"
if [[ "$STAGE" == all || "$STAGE" == plan || "$STAGE" == extract ]]; then
    python -u wan_offline_tools/extract_wan22_offline.py --all-train --frames 32 --plan-only | tee "$RECORD_DIR/feature_plan.log"
    if [[ "$STAGE" == plan ]]; then exit 0;fi
    PIDS=()
    cleanup_workers() { for pid in "${PIDS[@]}";do kill "$pid" 2>/dev/null || true;done; }
    trap cleanup_workers EXIT INT TERM
    for ((i=0;i<NUM_GPUS;i++)); do
        gpu="${GPU_IDS[i]}"
        python -u wan_offline_tools/extract_wan22_offline.py --all-train --frames 32 \
            --gpu "$gpu" --num-shards "$NUM_GPUS" --shard-index "$i" \
            > "$RECORD_DIR/extract_gpu${gpu}.log" 2>&1 &
        PIDS+=("$!")
        echo "[Qwen3.5] Extract worker GPU=$gpu PID=$! LOG=$RECORD_DIR/extract_gpu${gpu}.log"
    done
    EXTRACT_STATUS=0
    for pid in "${PIDS[@]}";do wait "$pid" || EXTRACT_STATUS=1;done
    if ((EXTRACT_STATUS));then
        for gpu in "${GPU_IDS[@]}";do tail -n 40 "$RECORD_DIR/extract_gpu${gpu}.log";done
        exit 1
    fi
    PIDS=();trap - EXIT INT TERM
    if [[ "$STAGE" == extract ]];then exit 0;fi
fi
if [[ "${QWEN35_FULL_PREFLIGHT:-0}" != 1 ]];then
    python -u wan_offline_tools/extract_wan22_offline.py --all-train --frames 32 --check-only | tee "$RECORD_DIR/feature_validation.log"
fi
if ((RESUME)); then
    compgen -G "$OUTPUT_DIR/checkpoint-*" >/dev/null || { echo '[Qwen3.5] No periodic checkpoint to resume';exit 1; }
else
    [[ ! -e "$OUTPUT_DIR" && ! -e "$LOG_FILE" ]] || { echo '[Qwen3.5] Output already exists. Use --resume for an interrupted run.';exit 1; }
fi
if [[ -n "${QWEN35_FULL_STORAGE_ROOT:-}" ]] && (( ! RESUME ));then
    mkdir -p "$QWEN35_FULL_STORAGE_ROOT/$RUN_NAME"
    STORAGE_TARGET="$(cd "$QWEN35_FULL_STORAGE_ROOT/$RUN_NAME" && pwd)"
    ln -s "$STORAGE_TARGET" "$OUTPUT_DIR"
    echo "[Qwen3.5] CHECKPOINT_STORAGE=$STORAGE_TARGET (accessible at $OUTPUT_DIR)"
fi
python - <<'RECORD'
from pathlib import Path
import os,json,hashlib,datetime
record=Path(os.environ['QWEN35_FULL_RECORD_DIR'])
names=['llava/train/train_3d.py','llava/train/llava_trainer.py','llava/model/llava_arch.py','llava/video_utils.py','scripts/3d/train/multi.yaml','scripts/zero3.json','scripts/3d/train/train_qwen35_wan22_full.sh']
names += ['wan_offline_tools/configure_wan_offline.py','wan_offline_tools/qwen35_memory_compat.py']
hashes={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in names}
p=record/'training_configuration.json'
data={'run_name':os.environ['QWEN35_FULL_RUN_NAME'],'gpus':os.environ['QWEN35_FULL_GPU_IDS'],'checkpoint':os.environ['QWEN35_FULL_OUTPUT_DIR'],'frames':32,'global_batch':int(os.environ.get('QWEN35_FULL_GLOBAL_BATCH','128')),'epochs':1,'max_steps':int(os.environ.get('QWEN35_FULL_MAX_STEPS','-1')),'context':32768,'save_steps':int(os.environ.get('QWEN35_FULL_SAVE_STEPS','500')),'torch_compile':os.environ.get('QWEN35_FULL_COMPILE','True'),'supervised_logits_only':os.environ.get('QWEN35_SUPERVISED_LOGITS_ONLY','0')=='1','original_source_hashes':hashes,'recorded_at':datetime.datetime.now().isoformat(),'changes_from_original':['Qwen3.5-2B-Language and native Qwen3.5 prompt/tokenizer','Frozen Wan2.2-TI2V-5B offline features, 1280x704 resize; reproducible scene seed/chunking as existing smoke',f"GPU count {len(os.environ['QWEN35_FULL_GPU_IDS'].split(','))} with accumulation {int(os.environ.get('QWEN35_FULL_GLOBAL_BATCH','128')) // len(os.environ['QWEN35_FULL_GPU_IDS'].split(','))} preserves global batch {int(os.environ.get('QWEN35_FULL_GLOBAL_BATCH','128'))}",'Original Trainer checkpoint call signature adapted in memory to installed Transformers','Output root vega3d_repro/checkpoints; no automatic post-train evaluation; profiling disabled and feature cache bounded']}
if not p.exists():p.write_text(json.dumps(data,ensure_ascii=False,indent=2))
else:
 old=json.loads(p.read_text())
 if old['original_source_hashes']!=hashes:raise RuntimeError('Original source changed since run began')
RECORD
IMAGE_FOLDER="data"
VIDEO_FOLDER="data"
FRAME_SAMPLING_STRATEGY="uniform"
PREV_STAGE_CHECKPOINT="data/models/Qwen3.5-2B-Language"
VISION_MODEL_VERSION="data/models/siglip-so400m-patch14-384"
PROMPT_VERSION="qwen35"
GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / NUM_GPUS))
GENERATIVE_PROFILE="${GENERATIVE_PROFILE:-True}"
export CUDA_VISIBLE_DEVICES="$GPUS"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export QWEN35_NUM_GPUS="$NUM_GPUS" QWEN35_FULL_FINAL_CHECKPOINT=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for path in "$PREV_STAGE_CHECKPOINT/config.json" "$VISION_MODEL_VERSION/config.json" "$DATA_YAML" scripts/zero3.json; do
    [[ -f "$path" ]] || { echo "[Qwen3.5] Missing file: $path" >&2; exit 1; }
done
if [[ "$FEATURE_SOURCE" == online ]]; then
    [[ -d data/models/Wan2.2-TI2V-5B && -f llava/model/multimodal_generative_encoder/wan22_prompt_embedding.pt ]] || { echo "[Qwen3.5] Missing online Wan checkpoint/prompt" >&2; exit 1; }
fi
python - <<'CHECK'
import json,os,torch
from transformers import AutoTokenizer
from llava import conversation as conversation_lib
assert torch.cuda.is_available(), 'CUDA unavailable'
assert torch.cuda.device_count()==int(os.environ['QWEN35_NUM_GPUS']), 'GPU visibility mismatch'
assert json.load(open('data/models/Qwen3.5-2B-Language/config.json'))['model_type']=='qwen3_5_text'
tok=AutoTokenizer.from_pretrained('data/models/Qwen3.5-2B-Language',local_files_only=True,use_fast=True)
assert tok.is_fast and tok.pad_token_id is not None
assert 'qwen35' in conversation_lib.conv_templates
for i in range(torch.cuda.device_count()):print('[Qwen3.5] GPU',i,torch.cuda.get_device_name(i))
CHECK
EVAL_ARG="$(python - <<'CHECK'
from transformers import TrainingArguments
fields=TrainingArguments.__dataclass_fields__
print('eval_strategy' if 'eval_strategy' in fields else 'evaluation_strategy')
CHECK
)"
mkdir -p "$OUTPUT_ROOT"
torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" --master_port "${MASTER_PORT:-43023}" \
    wan_offline_tools/configure_wan_offline.py --train-entry \
    --deepspeed scripts/zero3.json \
    --model_name_or_path "$PREV_STAGE_CHECKPOINT" \
    --version $PROMPT_VERSION \
    --data_path "$DATA_YAML" \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --embodiedscan_folder data/embodiedscan/ \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name "$RUN_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --max_steps "$MAX_STEPS" \
    --report_to none \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --"${EVAL_ARG}" "no" \
    --save_strategy "steps" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers "$DATA_WORKERS" \
    --lazy_preprocess True \
    "${COMPILE_ARGS[@]}" \
    --dataloader_drop_last True \
    --mm_newline_position grid \
    --add_spatial_instruction True \
    --force_sample True \
    --mm_spatial_pool_stride 2 \
    --world_position_embedding_type avg-discrete-sin3d \
    --object_feature_type patch14-pe \
    --ground_head_type infonce \
    --group_by_task_length True \
    --frame_sampling_strategy $FRAME_SAMPLING_STRATEGY \
    --frames_upbound "$FRAMES" \
    --use_generative_feature True \
    --generative_feat_dim 3072 \
    --generative_projector_type mlp2x_gelu \
    "${GEN_ARGS[@]}" \
    --generative_cache_max_items 1 \
    --generative_profile $GENERATIVE_PROFILE \
    --use_feat_fusion True \
    --feature_fusion_method token_gated \
    2>&1 | tee -a "$LOG_FILE"

python - "$OUTPUT_DIR" "$RECORD_DIR" <<'VERIFY'
import json,math,sys,hashlib
from pathlib import Path
run=Path(sys.argv[1]);record=Path(sys.argv[2])
state=json.loads((run/'trainer_state.json').read_text());config=json.loads((run/'config.json').read_text())
assert config['model_type']=='llava_qwen35'
assert state['global_step']>0
losses=[row['loss'] for row in state.get('log_history',[]) if 'loss' in row]
assert losses and all(math.isfinite(v) for v in losses)
assert (run/'model.safetensors').is_file() or (run/'model.safetensors.index.json').is_file()
original=json.loads((record/'training_configuration.json').read_text())['original_source_hashes']
for n,h in original.items():assert hashlib.sha256(Path(n).read_bytes()).hexdigest()==h,n
periodic=sorted(run.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
if periodic:
 p=periodic[-1]
 assert (p/'trainer_state.json').is_file()
 assert list(p.rglob('*optim_states.pt')), 'Missing DeepSpeed optimizer state'
audit={'run':str(run),'global_step':state['global_step'],'epoch':state.get('epoch'),'finite_logged_losses':True,'original_core_unchanged':True,'latest_periodic_checkpoint':str(periodic[-1]) if periodic else None,'final_model_saved':True}
(record/'completion_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
print('[Qwen3.5] FULL TRAIN SAVED',audit)
VERIFY
