import os
from pathlib import Path
from typing import Any

import joblib
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .config import FEATURE_COLUMNS, MODEL_FORMAT_VERSION, MODEL_PATH, RAW_DATA_DIR
from .data import build_provider_dataset


def train_and_save_model(
    raw_data_dir: Path = RAW_DATA_DIR,
    model_path: Path = MODEL_PATH,
    random_state: int = 42,
) -> dict[str, Any]:
    dataset = build_provider_dataset(raw_data_dir)
    if dataset["fraud"].nunique() != 2:
        raise ValueError("Training requires both No and Yes PotentialFraud provider labels.")
    if dataset["fraud"].value_counts().min() < 2:
        raise ValueError("Training requires at least two providers in each label class.")

    features = dataset.loc[:, list(FEATURE_COLUMNS)].astype("float64")
    target = dataset["fraud"].astype("int8")
    stratify = target if target.value_counts().min() >= 5 else None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=random_state,
        stratify=stratify,
    )

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
    metrics: dict[str, Any] = {
        "provider_count": int(len(dataset)),
        "train_provider_count": int(len(x_train)),
        "test_provider_count": int(len(x_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(y_test, probabilities))
            if y_test.nunique() == 2
            else None
        ),
    }

    bundle = {
        "format_version": MODEL_FORMAT_VERSION,
        "model": model,
        "feature_columns": list(FEATURE_COLUMNS),
        "threshold": 0.5,
        "metrics": metrics,
    }

    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        joblib.dump(bundle, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return bundle


def load_model_bundle(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {path}. Train it with: python -m fraud_detection.train"
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError("The model file has an unsupported format; retrain the model.")
    if bundle.get("format_version") != MODEL_FORMAT_VERSION:
        raise ValueError("The model file is from an unsupported version; retrain the model.")
    if bundle.get("feature_columns") != list(FEATURE_COLUMNS):
        raise ValueError("The model features do not match this application; retrain the model.")
    return bundle
