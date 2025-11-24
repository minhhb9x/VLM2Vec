from transformers import AutoConfig


MID = "apple/FastVLM-0.5B"
config = AutoConfig.from_pretrained(MID, trust_remote_code=True)
print(config)
