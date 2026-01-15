#!/bin/bash
# Example SFT training script
# Usage: bash scripts/sft/example_sft.sh

set -e
export PYTHONUNBUFFERED=1

OUTPUT_DIR=outputs/sft_example_lora_r16_$(date +%Y%m%d_%H%M%S)
mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info \
    accelerate launch \
    --main_process_port 29503 \
    --config_file scripts/accelerate/ds_zero2_4gpu.yaml \
    run.py sft \
    --config.common.seed 42 \
    --config.common.debug false \
    --config.model.model_name_or_path "Qwen/Qwen2.5-1.5B-Instruct" \
    --config.model.dtype "bfloat16" \
    --config.peft.use_peft true \
    --config.peft.type "lora" \
    --config.peft.r 16 \
    --config.peft.lora_alpha 32 \
    --config.peft.lora_dropout 0.05 \
    --config.peft.target_modules '["q_proj","v_proj","k_proj","o_proj","up_proj","down_proj","gate_proj"]' \
    --config.training.learning_rate 2e-5 \
    --config.training.output_dir "${OUTPUT_DIR}" \
    --config.training.run_name "${OUTPUT_DIR}" \
    --config.training.gradient_accumulation_steps 4 \
    --config.training.num_train_epochs 3 \
    --config.training.max_seq_length 512 \
    --config.training.packing false \
    --config.training.logging_steps 10 \
    --config.training.per_device_train_batch_size 2 \
    --config.training.save_strategy "steps" \
    --config.training.save_steps 24 \
    --config.training.warmup_ratio 0.03 \
    --config.training.lr_scheduler_type "cosine" \
    --config.training.bf16 true \
    --config.training.report_to '["wandb"]' \
    --config.logging.wandb_project "perl-sft" \
    --config.dataset.dataset_name_or_path "open-r1/DAPO-Math-17k-Processed" \
    2>&1 | tee "${OUTPUT_DIR}/output.log"
