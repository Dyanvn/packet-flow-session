import os
import time
import joblib
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.preprocessing.feature_selection import selected_features


DATASET_PATH = "data/processed/flow_model_dataset.csv"
OUTPUT_MODEL_PATH = "outputs/models/xgboost_flow_model.pkl"
OUTPUT_METRICS_PATH = "outputs/metrics/flow_training_metrics.csv"


def ensure_directories():
    os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_METRICS_PATH), exist_ok=True)


def load_dataset(path=DATASET_PATH):
    print(f"[INFO] Loading dataset from {path}...")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found. Run src/pipeline/build_flow_dataset.py first or create {path}."
        )
    df = pd.read_csv(path)
    X = df[selected_features]
    y = df["Label"]
    return X, y


def build_model(scale_pos_weight):
    return XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
    )


def evaluate(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def save_metrics(metrics):
    df = pd.DataFrame([metrics])
    df.to_csv(OUTPUT_METRICS_PATH, index=False)
    print(f"[INFO] Training metrics saved to {OUTPUT_METRICS_PATH}")


def main():
    ensure_directories()

    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\n========== DATA SHAPES ==========")
    print(f"X_train: {X_train.shape}")
    print(f"X_test : {X_test.shape}")

    attack_count = y_train.sum()
    benign_count = len(y_train) - attack_count
    scale_pos_weight = benign_count / attack_count

    print("\n========== CLASS INFO ==========")
    print(f"Benign : {benign_count}")
    print(f"Attack : {attack_count}")
    print(f"Scale Pos Weight: {scale_pos_weight:.4f}")

    model = build_model(scale_pos_weight)
    print("\n[INFO] Training XGBoost model...")
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start_time
    print(f"[INFO] Training completed in {training_time:.2f} seconds")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = evaluate(y_test, y_pred, y_prob)
    metrics.update(
        {
            "train_rows": X_train.shape[0],
            "test_rows": X_test.shape[0],
            "feature_count": X_train.shape[1],
            "training_time_seconds": round(training_time, 4),
        }
    )

    print("\n========== MODEL METRICS ==========")
    print(f"Accuracy : {metrics['accuracy']:.6f}")
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall   : {metrics['recall']:.6f}")
    print(f"F1 Score : {metrics['f1_score']:.6f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.6f}")

    joblib.dump(model, OUTPUT_MODEL_PATH)
    print(f"\n[INFO] Model saved to {OUTPUT_MODEL_PATH}")

    save_metrics(metrics)


if __name__ == "__main__":
    main()
