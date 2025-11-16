#!/bin/bash

# =========================================================
# 🔧 Basic configuration
# =========================================================

MODEL_NAME="apple/FastVLM-0.5B"
# MODEL_NAME="TIGER-Lab/VLM2Vec-Qwen2VL-2B"
CHECKPOINT_PATH="runs/FastVLM-0.5B/checkpoint-7500"
EXP_NAME="eval_small_model"
EXP_DIR="./exps/$EXP_NAME"

# Create base directories
mkdir -p "$EXP_DIR"

# =========================================================
# 🧠 Environment info
# =========================================================
echo "==> Environment"
echo "Python: $(which python)"
echo "Version: $(python --version)"
echo ""

# =========================================================
# 📂 Auto-generate output path based on model name
# =========================================================
# Replace slashes in model name with underscores
MODEL_ID=$(echo "$MODEL_NAME" | tr '/' '_')
ENCODE_DIR="$EXP_DIR/$MODEL_ID"

mkdir -p "$ENCODE_DIR"

echo "Model name: $MODEL_NAME"
echo "Encode output path: $ENCODE_DIR"
echo ""

# =========================================================
# 🚀 Run evaluation
# =========================================================
python eval.py \
  --model_name "$MODEL_NAME" \
  --pooling eos \
  --normalize true \
  --resize_use_processor false \
  --image_resolution high \
  --dataset_config scripts/eval/eval_image.yaml \
  --per_device_eval_batch_size 48 \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --encode_output_path "$ENCODE_DIR"

echo "✅ Evaluation completed. Results saved in: $ENCODE_DIR"
