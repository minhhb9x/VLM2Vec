#!/bin/bash

BASE_URL="https://huggingface.co/datasets/TIGER-Lab/MMEB-eval/resolve/main/images.zip"


DEST_DIR="./image-tasks"

set -e

wget "$BASE_URL"
unzip "images.zip" -d "$DEST_DIR"
rm "images.zip"

echo "*****************************************"
echo "TẤT CẢ CÁC FILE ĐÃ ĐƯỢC TẢI VÀ GIẢI NÉN!"
echo "*****************************************"

