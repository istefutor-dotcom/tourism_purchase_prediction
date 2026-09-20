import os
from pathlib import Path
from huggingface_hub import HfApi

TOKEN = os.environ["HF_TOKEN"]
SPACE_REPO = os.environ["HF_SPACE_REPO"]
MODEL_REPO = os.environ["HF_MODEL_REPO"]
DEPLOYMENT = Path("tourism_project/deployment")

readme = f'''---
title: Tourism Purchase Predictor
emoji: ✈️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---
# Tourism Purchase Predictor

This application loads the registered model from `{MODEL_REPO}`.
'''
(DEPLOYMENT / "README.md").write_text(readme)

api = HfApi(token=TOKEN)
api.create_repo(repo_id=SPACE_REPO, repo_type="space", space_sdk="docker", exist_ok=True)
api.upload_folder(
    folder_path=str(DEPLOYMENT),
    repo_id=SPACE_REPO,
    repo_type="space",
    ignore_patterns=["best_tourism_model.joblib", "push_to_space.py", "__pycache__/*"],
    commit_message="Deploy tourism prediction application",
)
api.add_space_variable(repo_id=SPACE_REPO, key="HF_MODEL_REPO", value=MODEL_REPO)
print(f"Space URL: https://huggingface.co/spaces/{SPACE_REPO}")
