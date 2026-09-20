import os
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from sklearn.model_selection import train_test_split

DATASET_REPO = os.environ["HF_DATASET_REPO"]
HF_TOKEN = os.environ["HF_TOKEN"]

raw_file = hf_hub_download(
    repo_id=DATASET_REPO,
    filename="tourism.csv",
    repo_type="dataset",
    token=HF_TOKEN,
)
data = pd.read_csv(raw_file)

columns_to_remove = [
    "Unnamed: 0",
    "CustomerID",
    "TypeofContact",
    "DurationOfPitch",
    "NumberOfFollowups",
    "ProductPitched",
    "PitchSatisfactionScore",
]
data = data.drop(columns=columns_to_remove, errors="ignore")

# Standardize known spelling variations.
data["Gender"] = data["Gender"].replace({"Fe Male": "Female"})
data["Occupation"] = data["Occupation"].replace({"Free Lancer": "Freelancer"})

if data["ProdTaken"].isna().any():
    raise ValueError("ProdTaken contains missing values.")
if set(data["ProdTaken"].unique()) - {0, 1}:
    raise ValueError("ProdTaken must contain only 0 and 1.")

# Arrange the thirteen predictors first and place the target last.
predictor_columns = [column for column in data.columns if column != "ProdTaken"]
data = data[predictor_columns + ["ProdTaken"]]

train_data, test_data = train_test_split(
    data,
    test_size=0.20,
    random_state=42,
    stratify=data["ProdTaken"],
)

# Learn all imputation values from the training data only, avoiding leakage.
numeric_columns = train_data[predictor_columns].select_dtypes(include="number").columns
categorical_columns = train_data[predictor_columns].select_dtypes(exclude="number").columns

for column in numeric_columns:
    training_median = train_data[column].median()
    train_data[column] = train_data[column].fillna(training_median)
    test_data[column] = test_data[column].fillna(training_median)

for column in categorical_columns:
    training_mode = train_data[column].mode().iloc[0]
    train_data[column] = train_data[column].fillna(training_mode)
    test_data[column] = test_data[column].fillna(training_mode)

output_directory = Path("tourism_project/data/processed")
output_directory.mkdir(parents=True, exist_ok=True)

train_path = output_directory / "train.csv"
test_path = output_directory / "test.csv"
train_data.to_csv(train_path, index=False)
test_data.to_csv(test_path, index=False)

api = HfApi(token=HF_TOKEN)
api.upload_folder(
    folder_path=str(output_directory),
    path_in_repo="processed",
    repo_id=DATASET_REPO,
    repo_type="dataset",
    commit_message="Create cleaned training and testing datasets",
)

print("Processed data uploaded successfully.")
print("Training shape:", train_data.shape)
print("Testing shape:", test_data.shape)
print("Training target percentages:")
print(train_data["ProdTaken"].value_counts(normalize=True).mul(100).round(2))
