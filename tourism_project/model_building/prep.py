import os
from pathlib import Path
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from sklearn.model_selection import train_test_split

DATASET_REPO = os.environ["HF_DATASET_REPO"]
HF_TOKEN = os.environ["HF_TOKEN"]

raw_file = hf_hub_download(
    repo_id=DATASET_REPO, filename="tourism.csv",
    repo_type="dataset", token=HF_TOKEN
)
df = pd.read_csv(raw_file)

# CustomerID is an identifier and is not a predictive feature.
df = df.drop(columns=["CustomerID"])
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

output_dir = Path("tourism_project/data/prepared")
output_dir.mkdir(parents=True, exist_ok=True)
files = {
    "Xtrain.csv": Xtrain, "Xtest.csv": Xtest,
    "ytrain.csv": ytrain.to_frame(), "ytest.csv": ytest.to_frame(),
}
for filename, frame in files.items():
    local_path = output_dir / filename
    frame.to_csv(local_path, index=False)

api = HfApi(token=HF_TOKEN)
api.upload_folder(
    folder_path=str(output_dir), path_in_repo="prepared",
    repo_id=DATASET_REPO, repo_type="dataset"
)
print("Prepared data uploaded successfully.")
print("Training shape:", Xtrain.shape, "Test shape:", Xtest.shape)
print("Training target distribution:\n", ytrain.value_counts(normalize=True))
