#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
REPO_ROOT="$(pwd)"
QWEN35_ENV_ROOT="${QWEN35_ENV_ROOT:-$(dirname "$(dirname "$REPO_ROOT")")/envs/vega3d-qwen35}"
export PATH="$QWEN35_ENV_ROOT/bin:$PATH"
MODE="smoke"
FEATURE_SOURCE="offline"
GPUS="2,3"
OUTPUT_ROOT="$(dirname "$(dirname "$REPO_ROOT")")/checkpoints"
RUN_NAME=""
DATA_YAML="scripts/3d/train/multi.yaml"
MAX_STEPS="10"
NUM_TRAIN_EPOCHS="1"
FRAMES="8"
FEATURE_ID="auto"
DRY_RUN=0

usage() {
    cat <<'HELP'
Qwen3.5 + Wan unified entry point
Usage: bash scripts/3d/train/train_qwen35_wan22.sh [options]
  --mode smoke|train|eval    Default: complete small offline smoke
  --gpus 2,3                Visible physical GPUs; default 2,3
  --feature-source offline|online
  --run-name NAME           Names always receive the qwen35- prefix
  --output-root PATH        Default: vega3d_repro/checkpoints (absolute path)
  --data-path PATH          Training data JSON or YAML (train mode)
  --max-steps N             Default train limit: 10; -1 for epoch-based training
  --epochs N                Default: 1
  --frames N                Default: 8; offline frames must match extraction
  --feature-id ID           Default: read the existing Wan manifest
  --dry-run                 Print the selected command without running it
  --help

Examples:
  bash scripts/3d/train/train_qwen35_wan22.sh
  bash scripts/3d/train/train_qwen35_wan22.sh --mode smoke --gpus 4,5
  bash scripts/3d/train/train_qwen35_wan22.sh --mode train --feature-source online --gpus 2 --max-steps 10
  bash scripts/3d/train/train_qwen35_wan22.sh --mode eval --run-name qwen35-wan22-example

Training uses the original train_3d.py and original sampler. Saved final model,
logs and run names explicitly identify Qwen3. No optimizer-resume state is
promised: periodic checkpoint saving is disabled for this smoke entry point.
Full offline training requires features for all scenes in --data-path.
HELP
}

while (($#)); do
    case "$1" in
        --mode|--gpus|--feature-source|--run-name|--output-root|--data-path|--max-steps|--epochs|--frames|--feature-id)
            [[ $# -ge 2 ]] || { echo "[Qwen3.5] Missing value: $1" >&2; exit 2; }
            case "$1" in
                --mode) MODE="$2";;
                --gpus) GPUS="$2";;
                --feature-source) FEATURE_SOURCE="$2";;
                --run-name) RUN_NAME="$2";;
                --output-root) OUTPUT_ROOT="$2";;
                --data-path) DATA_YAML="$2";;
                --max-steps) MAX_STEPS="$2";;
                --epochs) NUM_TRAIN_EPOCHS="$2";;
                --frames) FRAMES="$2";;
                --feature-id) FEATURE_ID="$2";;
            esac
            shift 2;;
        --dry-run) DRY_RUN=1; shift;;
        --help|-h) usage; exit 0;;
        *) echo "[Qwen3.5] Unknown option: $1" >&2; usage; exit 2;;
    esac
done
[[ "$MODE" =~ ^(smoke|train|eval)$ ]] || { echo "[Qwen3.5] Invalid mode" >&2; exit 2; }
[[ "$FEATURE_SOURCE" =~ ^(offline|online)$ ]] || { echo "[Qwen3.5] Invalid feature source" >&2; exit 2; }
[[ "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "[Qwen3.5] Invalid GPU list" >&2; exit 2; }
[[ "$MAX_STEPS" =~ ^(-1|[1-9][0-9]*)$ ]] || { echo "[Qwen3.5] --max-steps must be -1 or positive" >&2; exit 2; }
[[ "$NUM_TRAIN_EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "[Qwen3.5] --epochs must be positive" >&2; exit 2; }
[[ "$FRAMES" =~ ^[1-9][0-9]*$ ]] || { echo "[Qwen3.5] --frames must be positive" >&2; exit 2; }
IFS=',' read -r -a GPU_IDS <<< "$GPUS"
for ((i=0;i<${#GPU_IDS[@]};i++)); do
    for ((j=i+1;j<${#GPU_IDS[@]};j++)); do
        [[ "${GPU_IDS[i]}" != "${GPU_IDS[j]}" ]] || { echo "[Qwen3.5] Duplicate GPU index" >&2; exit 2; }
    done
done
NUM_GPUS="${#GPU_IDS[@]}"
if [[ -n "$RUN_NAME" ]]; then
    [[ "$RUN_NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "[Qwen3.5] Invalid run name" >&2; exit 2; }
    [[ "$RUN_NAME" == qwen35-* ]] || RUN_NAME="qwen35-$RUN_NAME"
fi

if [[ "$MODE" == smoke || "$MODE" == eval ]]; then
    [[ "$FEATURE_SOURCE" == offline ]] || { echo "[Qwen3.5] Complete smoke/eval uses verified offline Wan features; online is available in train mode." >&2; exit 2; }
    [[ "$FRAMES" == 8 ]] || { echo "[Qwen3.5] This smoke plan uses 8 frames." >&2; exit 2; }
    [[ "$MODE" != eval || -n "$RUN_NAME" ]] || { echo "[Qwen3.5] --run-name is required for eval." >&2; exit 2; }
    STAGE="all"
    [[ "$MODE" != eval ]] || STAGE="eval"
    CMD=(python wan_full_smoke_tools/run_full_smoke.py --stage "$STAGE" --gpus "$GPUS" --output-root "$OUTPUT_ROOT")
    [[ -z "$RUN_NAME" ]] || CMD+=(--run-name "$RUN_NAME")
    echo "[Qwen3.5] MODE=$MODE GPUs=$GPUS OUTPUT_ROOT=$OUTPUT_ROOT"
    if ((DRY_RUN)); then printf '[Qwen3.5] Command: '; printf '%q ' "${CMD[@]}"; printf '\n'; exit 0; fi
    exec "${CMD[@]}"
fi

if [[ -z "$RUN_NAME" ]]; then
    STAMP="$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)"
    RUN_NAME="qwen35-wan22-${FEATURE_SOURCE}-train-gpu${GPUS//,/}-${STAMP}-$$"
fi
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_NAME"
LOG_FILE="$OUTPUT_ROOT/$RUN_NAME.log"
GEN_ARGS=(--generative_feature_source "$FEATURE_SOURCE")
if [[ "$FEATURE_SOURCE" == online ]]; then
    GEN_ARGS+=(--generative_vision_tower_type wan22_ti2v_online
        --generative_vision_tower_checkpoint data/models/Wan2.2-TI2V-5B
        --generative_vision_tower_task ti2v-5B
        --generative_vision_tower_size '1280*704'
        --generative_vision_tower_timestep 300 --generative_vision_tower_shift 5.0
        --generative_vision_tower_feat_block_idx 20
        --generative_vision_tower_prompt_emb_path llava/model/multimodal_generative_encoder/wan22_prompt_embedding.pt)
else
    if [[ "$FEATURE_ID" == auto ]]; then
        FEATURE_ID="$(python - <<'FEATURE'
import json
from pathlib import Path
pointer=json.loads(Path('wan_offline_tools/generated/qwen35_wan_offline_manifest.json').read_text())
print(json.loads(Path(pointer['manifest']).read_text())['feature_id'])
FEATURE
)"
    fi
    [[ "$FEATURE_ID" =~ ^wan22_ti2v_offline_uniform[0-9]+_[0-9]+x[0-9]+_t[0-9]+_b[0-9]+_s[0-9]+_v1$ ]] || { echo "[Qwen3.5] Invalid offline feature ID" >&2; exit 2; }
    [[ "$FEATURE_ID" == wan22_ti2v_offline_uniform${FRAMES}_* ]] || { echo "[Qwen3.5] Offline features do not match --frames" >&2; exit 2; }
    GEN_ARGS+=(--generative_model_id "$FEATURE_ID")
fi

echo "[Qwen3.5] MODEL=data/models/Qwen3.5-2B-Language MODE=train GPUs=$GPUS WAN=$FEATURE_SOURCE"
echo "[Qwen3.5] RUN_NAME=$RUN_NAME"
echo "[Qwen3.5] CHECKPOINT=$OUTPUT_DIR"
echo "[Qwen3.5] LOG=$LOG_FILE"
if ((DRY_RUN)); then
    echo "[Qwen3.5] DATA=$DATA_YAML MAX_STEPS=$MAX_STEPS EPOCHS=$NUM_TRAIN_EPOCHS FRAMES=$FRAMES"
    printf '[Qwen3.5] Generative arguments: '; printf '%q ' "${GEN_ARGS[@]}"; printf '\n'
    exit 0
fi
[[ ! -e "$OUTPUT_DIR" && ! -e "$LOG_FILE" ]] || { echo "[Qwen3.5] Run already exists; choose a new --run-name." >&2; exit 2; }

IMAGE_FOLDER="data"
VIDEO_FOLDER="data"
FRAME_SAMPLING_STRATEGY="uniform"
PREV_STAGE_CHECKPOINT="data/models/Qwen3.5-2B-Language"
VISION_MODEL_VERSION="data/models/siglip-so400m-patch14-384"
PROMPT_VERSION="qwen35"
GRADIENT_ACCUMULATION_STEPS=1
GENERATIVE_PROFILE="${GENERATIVE_PROFILE:-True}"
export CUDA_VISIBLE_DEVICES="$GPUS"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export QWEN35_NUM_GPUS="$NUM_GPUS"

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
    llava/train/train_3d.py \
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
    --save_strategy "no" \
    --save_steps 10 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --torch_compile False \
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
    2>&1 | tee "$LOG_FILE"

python - "$OUTPUT_DIR" <<'VERIFY'
import json,math,sys
from pathlib import Path
run=Path(sys.argv[1]); state=json.loads((run/'trainer_state.json').read_text())
config=json.loads((run/'config.json').read_text())
assert config['model_type']=='llava_qwen35', 'Saved model is not VEGA-Qwen3.5'
assert state['global_step']>0, 'No training steps completed'
losses=[row['loss'] for row in state.get('log_history',[]) if 'loss' in row]
assert losses and all(math.isfinite(v) for v in losses), 'Invalid training losses'
assert (run/'model.safetensors').is_file() or (run/'model.safetensors.index.json').is_file(), 'Missing weights'
print('[Qwen3.5] Training saved:',run,'actual steps:',state['global_step'])
print('[Qwen3.5] Final model weights saved; optimizer-resume state is not included.')
VERIFY
