#!/bin/bash

BASE_URL="https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip"


DEST_DIR="./vlm2vec_train/MMEB-train/images"

# Danh sách tất cả các file .zip cần tải
FILES=(
    "A-OKVQA.zip"
    "CIRR.zip"
    "ChartQA.zip"
    "DocVQA.zip"
    "HatefulMemes.zip"
    "ImageNet_1K.zip"
    "InfographicsVQA.zip"
    "MSCOCO.zip"
    "MSCOCO_i2t.zip"
    "MSCOCO_t2i.zip"
    "N24News.zip"
    "NIGHTS.zip"
    "OK-VQA.zip"
    "SUN397.zip"
    "VOC2007.zip"
    "VisDial.zip"
    "Visual7W.zip"
    "VisualNews_i2t.zip"
    "VisualNews_t2i.zip"
    "WebQA.zip"
)
# --- KẾT THÚC CẤU HÌNH ---

# -------------------------------------------------
# set -e: Thoát script ngay nếu có bất kỳ lệnh nào lỗi
set -e

# 1. Tạo thư mục đích (nếu chưa tồn tại)
echo "Tạo thư mục đích tại: $DEST_DIR"
mkdir -p "$DEST_DIR"

# 2. Vòng lặp qua từng file trong danh sách
for file in "${FILES[@]}"; do
    echo "========================================="
    echo "BẮT ĐẦU xử lý file: $file"
    echo "========================================="

    # 2a. Tải file
    echo "Đang tải: $BASE_URL/$file"
    wget "$BASE_URL/$file"

    # 2b. Giải nén vào thư mục đích
    echo "Đang giải nén $file vào $DEST_DIR"
    unzip "$file" -d "$DEST_DIR"

    # 2c. Xóa file zip đã tải
    echo "Đang xóa file: $file"
    rm "$file"

    echo ">>> HOÀN THÀNH xử lý file: $file"
    echo "" # Thêm một dòng trống cho dễ đọc

done

echo "*****************************************"
echo "TẤT CẢ CÁC FILE ĐÃ ĐƯỢC TẢI VÀ GIẢI NÉN!"
echo "*****************************************"