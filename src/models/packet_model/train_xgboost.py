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


DATASET_PATH = "data/processed/packet_model_dataset.csv"
OUTPUT_MODEL_PATH = "outputs/models/xgboost_packet_model.pkl"
OUTPUT_METRICS_PATH = "outputs/metrics/packet_training_metrics.csv"

SELECTED_FEATURES = [
    "Timestamp",
    "Src Port",
    "Dst Port",
    "Packet Length",
    "SYN Flag",
    "ACK Flag",
    "FIN Flag",
    "RST Flag",
    "PSH Flag",
    "URG Flag",
    "Protocol Code",
]

PROTOCOL_MAP = {"TCP": 1, "UDP": 0}


def ensure_directories():
    os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_METRICS_PATH), exist_ok=True)


def load_dataset(path=DATASET_PATH):
    print(f"[INFO] Loading packet dataset from {path}...")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Packet dataset not found. Run src/pipeline/build_packet_dataset.py first or create {path}."
        )
    df = pd.read_csv(path)
    df = df.dropna(subset=["Label"])
    if "Protocol" in df.columns:
        df["Protocol Code"] = df["Protocol"].map(PROTOCOL_MAP).fillna(2).astype(int)
    else:
        df["Protocol Code"] = 2
    missing = [feat for feat in SELECTED_FEATURES if feat not in df.columns]
    if missing:
        raise ValueError(f"Missing packet features: {missing}")
    X = df[SELECTED_FEATURES]
    y = df["Label"].astype(int)
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
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def save_metrics(metrics):
    df = pd.DataFrame([metrics])
    df.to_csv(OUTPUT_METRICS_PATH, index=False)
    print(f"[INFO] Packet training metrics saved to {OUTPUT_METRICS_PATH}")


def main():
    ensure_directories()
    X, y = load_dataset()
    if len(y.unique()) < 2:
        raise ValueError(
            "Packet dataset must contain at least two label classes for training."
        )

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
    scale_pos_weight = benign_count / attack_count if attack_count > 0 else 1.0

    print("\n========== CLASS INFO ==========")
    print(f"Benign : {benign_count}")
    print(f"Attack : {attack_count}")
    print(f"Scale Pos Weight: {scale_pos_weight:.4f}")

    model = build_model(scale_pos_weight)
    print("\n[INFO] Training packet-level XGBoost model...")
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

    print("\n========== PACKET MODEL METRICS ==========")
    print(f"Accuracy : {metrics['accuracy']:.6f}")
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall   : {metrics['recall']:.6f}")
    print(f"F1 Score : {metrics['f1_score']:.6f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.6f}")

    joblib.dump(model, OUTPUT_MODEL_PATH)
    print(f"\n[INFO] Packet model saved to {OUTPUT_MODEL_PATH}")
    save_metrics(metrics)


if __name__ == "__main__":
    main()
