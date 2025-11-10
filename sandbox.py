import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

MID = "apple/FastVLM-0.5B"
IMAGE_TOKEN_INDEX = -200  # what the model code looks for

# Load
tok = AutoTokenizer.from_pretrained(MID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)

print(type(model))
print(type(tok))
# Build chat -> render to string (not tokens) so we can place <image> exactly
config = AutoConfig.from_pretrained(MID, trust_remote_code=True)

# In ra tên model type
print("Model type:", config.model_type)

# In toàn bộ config (optional)
# print(config)

