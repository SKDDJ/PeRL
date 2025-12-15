#! /bin/bash

set -exo pipefail
ulimit -n 65535

PROJECT_DIR="."
BASE_MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# DATASET="aime2024@512,aime2025@512,amc2023@32,math500@8,minerva@8,hmmt2025@32"
DATASET="aime2024@32,aime2025@32,amc2023@32,math500@4,minerva@4,hmmt2025@32"
# DATASET="aime2024@512,aime2025@512" # for test

export PYTHONPATH="${PROJECT_DIR}"
export HF_ENDPOINT="https://hf-mirror.com"
export VLLM_TORCH_COMPILE="0"

TEMPERATURE="0.7"
TOP_P="0.9"
MAX_NEW_TOKENS="31744"
# MAX_NEW_TOKENS="65536"
CUDA_VISIBLE_DEVICES=0,1,2,3
DP_SIZE=4
TP_SIZE=1
MAX_NUM_REQUEST="$((200 * ${DP_SIZE}))"
GPU_MEMORY_UTILIZATION=0.95

function kill_vllm_processes() {
  pkill -9 -f "vllm.entrypoints.openai.api_server" || true;
  pkill -9 -f "VLLM::EngineCore" || true;
  sleep 1;
  pkill -9 -f "vllm.entrypoints.openai.api_server" || true;
  pkill -9 -f "VLLM::EngineCore" || true;
}

function eval_model_with_adapter() {
  kill_vllm_processes;
  
  RESULT_DIR="$1"
  MODEL_DIR="$2"
  ADAPTER_DIR="$3"

  mkdir -p "${RESULT_DIR}"
  
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python "${PROJECT_DIR}/perl/eval.py" \
    --prompt-format "open-r1" \
    --result-dir "${RESULT_DIR}" \
    --model "${MODEL_DIR}" \
    --adapter "${ADAPTER_DIR}" \
    --dataset "${DATASET}" \
    --serve-port 8000 \
    --dp-size "${DP_SIZE}" \
    --tp-size "${TP_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --seed "42" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --max-num-request "${MAX_NUM_REQUEST}" \
    --dtype "bfloat16" 2>&1 | tee "eval.log";
}

function eval_model_with_adapter_from_hf() {
  mkdir -p "${PROJECT_DIR}/ckpts/$1/$2";

  hf download MikaStars39/PeRL \
      --repo-type model \
      --local-dir "${PROJECT_DIR}/ckpts/" \
      --include "$1/$2/*";

  ls -a "${PROJECT_DIR}/ckpts/$1/$2";

  eval_model_with_adapter \
      "${PROJECT_DIR}/outputs/eval/openr1-$1___$2" \
      "${BASE_MODEL_PATH}" \
      "${PROJECT_DIR}/ckpts/$1/$2";
}

set +e

eval_model_with_adapter \
   "${PROJECT_DIR}/outputs/grpo_adalora_qwen2_5_1_5b_20251212_181345" \
   "${BASE_MODEL_PATH}" \
   "${PROJECT_DIR}/outputs/grpo_adalora_qwen2_5_1_5b_20251212_181345"

