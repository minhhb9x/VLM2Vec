from huggingface_hub import HfApi, Repository, upload_folder
import os
from dotenv import load_dotenv
load_dotenv()

# --- Cấu hình ---
MODEL_DIR = "./runs/FastVLM-0.5B/checkpoint-7500"      # folder model local sau khi train xong
HF_USERNAME = "minhhotboy9x"     # username Hugging Face của bạn
MODEL_NAME = "FASTVLM-0.5B-checkpoint-7500"      # tên repo trên HF
PRIVATE = False                    # True nếu muốn repo private
TOKEN = os.getenv("HF_TOKEN")               # token Hugging Face của bạn

# --- Tạo repo trên HF (nếu chưa có) ---
api = HfApi()
# try:
#     api.create_repo(
#         repo_id=f"{HF_USERNAME}/{MODEL_NAME}",
#         token=TOKEN,
#         private=PRIVATE
#     )
#     print(f"Repo '{MODEL_NAME}' created successfully!")
# except Exception as e:
#     print(f"Repo creation skipped or failed: {e}")

# # --- Clone repo và copy file ---
# repo_url = f"https://huggingface.co/{HF_USERNAME}/{MODEL_NAME}"
# repo = Repository(local_dir=MODEL_DIR+"_repo", clone_from=repo_url, use_auth_token=TOKEN)

# # Copy tất cả file model vào folder repo local
# import shutil
# for file_name in os.listdir(MODEL_DIR):
#     full_file_name = os.path.join(MODEL_DIR, file_name)
#     if os.path.isfile(full_file_name):
#         shutil.copy(full_file_name, repo.local_dir)

# model_card_content = f"""
# # {MODEL_NAME}

# ## Model description
# This is a trained VLM embedding model from FastVLM-0.5B at checkpoint 7500.

# """
# with open(os.path.join(repo.local_dir, "README.md"), "w", encoding="utf-8") as f:
#     f.write(model_card_content)

# # --- Commit & push ---
# repo.push_to_hub(commit_message="Upload trained model")
# print("Model pushed to Hugging Face Hub successfully!")

api.upload_folder(
    folder_path="exps/eval_small_model/apple_FastVLM-0.5B",    
    repo_id="minhhotboy9x/FASTVLM-0.5B-checkpoint-7500",    
    repo_type="model",                     
    path_in_repo="eval_image",         
    token=TOKEN,
)
