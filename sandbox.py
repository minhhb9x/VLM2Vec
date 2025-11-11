from transformers import AutoConfig
config = AutoConfig.from_pretrained("apple/FastVLM-0.5B", trust_remote_code=True)
print(config.mm_vision_tower)
print(config.mm_projector_lr)