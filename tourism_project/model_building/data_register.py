
import os

import pandas as pd
from huggingface_hub import HfApi


# Configuration
RAW_DATA_PATH = "tourism_project/data/tourism.csv"
DATASET_REPO = os.environ["HF_DATASET_REPO"]
HF_TOKEN = os.environ["HF_TOKEN"]


# Load the local dataset
df = pd.read_csv(RAW_DATA_PATH)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# Validate the important columns
required_columns = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )

print("Dataset column validation completed.")


# Create the Hugging Face Dataset repository
api = HfApi(token=HF_TOKEN)

api.create_repo(
    repo_id=DATASET_REPO,
    repo_type="dataset",
    exist_ok=True
)

print("Hugging Face Dataset repository is available.")


# Upload the original dataset
api.upload_file(
    path_or_fileobj=RAW_DATA_PATH,
    path_in_repo="tourism.csv",
    repo_id=DATASET_REPO,
    repo_type="dataset",
    commit_message="Register original tourism dataset"
)

print("Dataset registered successfully.")
print(
    f"Dataset URL: "
    f"https://huggingface.co/datasets/{DATASET_REPO}"
)
