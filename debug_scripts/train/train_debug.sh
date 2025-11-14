#!/bin/bash

# === Minimal debug version (1 GPU) ===

export EXP_NAME=debug_model
export EXP_DIR=./runs/$EXP_NAME
mkdir -p $EXP_DIR

echo "Python: $(which python)"
echo "Version: $(python --version)"


# --model_name Qwen/Qwen2-VL-2B-Instruct \

python train.py \
  --lora --lora_r 16 \
  --model_name apple/FastVLM-0.5B \
  --bf16 --pooling eos --normalize True --temperature 0.02 \
  --dataset_config debug_scripts/train/train_image_debug.yaml \
  --run_name $EXP_NAME --output_dir $EXP_DIR \
  --learning_rate 5e-5 \
  --grad_cache False \
  --per_device_train_batch_size 16 \
  --gc_q_chunk_size 2 \
  --gc_p_chunk_size 2 \
  --interleave_batch_size 4 \
  --lr_scheduler_type linear \
  --warmup_steps 5 \
  --max_steps 50 \
  --save_steps 10 \
  --logging_steps 1 \
  --save_safetensors True \
  --remove_unused_columns False \
  --report_to none 2>&1 | tee $EXP_DIR/train.log
