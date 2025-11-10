import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from pprint import pprint  # để in đẹp hơn

MID = "apple/FastVLM-0.5B"

# Hằng số mô phỏng vị trí ảnh
IMAGE_TOKEN_INDEX = -200  

# 1️⃣ Load tokenizer
tok = AutoTokenizer.from_pretrained(MID, trust_remote_code=True)

# 2️⃣ Load model
model = AutoModelForCausalLM.from_pretrained(
    MID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)

# 3️⃣ Load config
config = AutoConfig.from_pretrained(MID, trust_remote_code=True)

# 4️⃣ In thông tin tổng quan
print("=" * 60)
print("✅ MODEL INFORMATION")
print(f"Model class: {type(model)}")
print(f"Tokenizer class: {type(tok)}")
print(f"Model type: {config.model_type}")
print(f"Padding side: {getattr(config, 'padding_side', 'N/A')}")
print("=" * 60)

# 5️⃣ In cấu hình chi tiết (rút gọn)
print("\n🧠 CONFIG DETAILS (rút gọn):")
keys_to_show = [
    "model_type", "architectures", "torch_dtype",
    "vision_tower", "image_size", "patch_size",
    "text_config", "vision_config",
    "use_cache", "padding_side", "tie_word_embeddings"
]

for key in keys_to_show:
    value = getattr(config, key, None)
    if value is not None:
        print(f"- {key}: {value}")

# 6️⃣ Nếu bạn muốn xem toàn bộ config (không rút gọn):
# print("\nFull config dump:")
# pprint(config.to_dict())
