import os
from PIL import Image

root_dir = "vlm2vec_train/MMEB-train/images"
min_size = 10  # threshold cho ảnh quá nhỏ

small_images = []

for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            path = os.path.join(dirpath, filename)
            try:
                with Image.open(path) as img:
                    w, h = img.size
                    if w < min_size or h < min_size:
                        small_images.append((path, w, h))
            except Exception as e:
                print(f"Could not open {path}: {e}")

print(f"Found {len(small_images)} small images:")
for path, w, h in small_images:
    print(path, w, h)
