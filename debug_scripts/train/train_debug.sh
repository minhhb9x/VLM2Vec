#!/bin/bash

# === Minimal debug version (1 GPU) ===

export EXP_NAME=debug_model
export EXP_DIR=./runs/$EXP_NAME
mkdir -p $EXP_DIR

echo "Python: $(which python)"
echo "Version: $(python --version)"

torchrun --nproc_per_node=1 train.py \
  --lora --lora_r 16 \
  --model_name Qwen/Qwen2-VL-2B-Instruct \
  --bf16 --pooling eos --normalize True --temperature 0.02 \
  --dataset_config debug_scripts/train/train_image_debug.yaml \
  --run_name $EXP_NAME --output_dir $EXP_DIR \
  --per_device_train_batch_size 4 \
  --learning_rate 5e-5 --max_steps 50 \
  --save_steps 10 --logging_steps 1 \
  --report_to none 2>&1 | tee $EXP_DIR/train.log
