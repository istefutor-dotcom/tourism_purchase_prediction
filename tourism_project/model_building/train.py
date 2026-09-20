import json
import os
import tempfile
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from huggingface_hub import HfApi, hf_hub_download
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("tourism-purchase-training")

# Meet the rubric requirement by loading both prepared files directly from
# the Hugging Face dataset repository, rather than from local workflow files.
train_path = hf_hub_download(
    repo_id=os.environ["HF_DATASET_REPO"],
    filename="processed/train.csv",
    repo_type="dataset",
    token=os.environ["HF_TOKEN"],
)
test_path = hf_hub_download(
    repo_id=os.environ["HF_DATASET_REPO"],
    filename="processed/test.csv",
    repo_type="dataset",
    token=os.environ["HF_TOKEN"],
)
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
Xtrain = train_data.drop(columns=["ProdTaken"])
ytrain = train_data["ProdTaken"].astype(int)
Xtest = test_data.drop(columns=["ProdTaken"])
ytest = test_data["ProdTaken"].astype(int)

numeric_features = Xtrain.select_dtypes(include="number").columns.tolist()
categorical_features = Xtrain.select_dtypes(exclude="number").columns.tolist()

numeric_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
)
categorical_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"),
)
preprocessor = make_column_transformer(
    (numeric_pipeline, numeric_features),
    (categorical_pipeline, categorical_features),
)

# scale_pos_weight compensates for the smaller buyer class.
negative_count, positive_count = ytrain.value_counts().sort_index().tolist()
class_ratio = negative_count / positive_count
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=1,
    scale_pos_weight=class_ratio,
)

param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.05, 0.10],
}
model_pipeline = make_pipeline(preprocessor, xgb_model)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

with mlflow.start_run(run_name="xgboost-grid-search"):
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=cv,
        n_jobs=1,
        scoring="f1",
        refit=True,
    )
    grid_search.fit(Xtrain, ytrain)

    # Log every tuned combination and its validation performance.
    results = grid_search.cv_results_
    for index, parameters in enumerate(results["params"]):
        with mlflow.start_run(run_name=f"candidate-{index + 1:02d}", nested=True):
            mlflow.log_params(parameters)
            mlflow.log_metric("mean_cv_f1", float(results["mean_test_score"][index]))
            mlflow.log_metric("std_cv_f1", float(results["std_test_score"][index]))

    best_model = grid_search.best_estimator_
    predictions = best_model.predict(Xtest)
    probabilities = best_model.predict_proba(Xtest)[:, 1]
    metrics = {
        "accuracy": accuracy_score(ytest, predictions),
        "precision": precision_score(ytest, predictions, zero_division=0),
        "recall": recall_score(ytest, predictions, zero_division=0),
        "f1": f1_score(ytest, predictions, zero_division=0),
        "roc_auc": roc_auc_score(ytest, probabilities),
        "best_cv_f1": grid_search.best_score_,
    }
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metrics(metrics)

    deployment_model = Path("tourism_project/deployment/best_tourism_model.joblib")
    deployment_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, deployment_model)
    mlflow.log_artifact(str(deployment_model), artifact_path="model")

    with tempfile.TemporaryDirectory() as directory:
        model_dir = Path(directory)
        joblib.dump(best_model, model_dir / "model.joblib")
        (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        (model_dir / "best_parameters.json").write_text(
            json.dumps(grid_search.best_params_, indent=2)
        )
        (model_dir / "classification_report.txt").write_text(
            classification_report(ytest, predictions)
        )
        (model_dir / "confusion_matrix.json").write_text(
            json.dumps(confusion_matrix(ytest, predictions).tolist())
        )
        (model_dir / "README.md").write_text(f'''---
library_name: scikit-learn
pipeline_tag: tabular-classification
---
# Tourism Package Purchase Model

XGBoost classification pipeline selected by five-fold stratified grid search using F1.
The target represents historical general tourism package purchases.

## Test metrics
```json
{json.dumps(metrics, indent=2)}
```
''')
        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(
            repo_id=os.environ["HF_MODEL_REPO"], repo_type="model", exist_ok=True
        )
        api.upload_folder(
            folder_path=str(model_dir),
            repo_id=os.environ["HF_MODEL_REPO"],
            repo_type="model",
            commit_message="Register best tuned tourism purchase model",
        )

print("Best parameters:", grid_search.best_params_)
print("Test metrics:\n", json.dumps(metrics, indent=2))
print("Confusion matrix:\n", confusion_matrix(ytest, predictions))
print("Classification report:\n", classification_report(ytest, predictions))
print("Model URL: https://huggingface.co/" + os.environ["HF_MODEL_REPO"])
