from transformers import AutoConfig


FASTVLM = "apple/FastVLM-0.5B"
config = AutoConfig.from_pretrained(FASTVLM, trust_remote_code=True)
print(config)


QWEN2VL = "TIGER-Lab/VLM2Vec-Qwen2VL-2B"
config = AutoConfig.from_pretrained(QWEN2VL, trust_remote_code=True)
print(config)