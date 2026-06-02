import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split


DATASET_PATH = "data/processed/packet_model_dataset.csv"
OUTPUT_METRICS_PATH = "outputs/metrics/packet_evaluation_metrics.csv"
OUTPUT_CONFUSION_PLOT = "outputs/figures/packet_confusion_matrix.png"
OUTPUT_ROC_PLOT = "outputs/figures/packet_roc_curve.png"
OUTPUT_METRICS_TABLE = "outputs/metrics/packet_classification_report.csv"

PROTOCOL_MAP = {"TCP": 1, "UDP": 0}
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


def ensure_directories():
    os.makedirs(os.path.dirname(OUTPUT_METRICS_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_CONFUSION_PLOT), exist_ok=True)


def load_dataset(path=DATASET_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Packet dataset not found. Run src/pipeline/build_packet_dataset.py first or create {path}."
        )
    df = pd.read_csv(path)
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


def save_classification_report(report, path):
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(path, index=True)
    print(f"[INFO] Packet classification report saved to {path}")


def save_metrics(metrics, path):
    df = pd.DataFrame([metrics])
    df.to_csv(path, index=False)
    print(f"[INFO] Packet evaluation metrics saved to {path}")


def main():
    ensure_directories()
    X, y = load_dataset()
    if len(y.unique()) < 2:
        raise ValueError(
            "Packet dataset must contain at least two label classes for evaluation."
        )

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = joblib.load("outputs/models/xgboost_packet_model.pkl")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, digits=6, output_dict=True)
    print(report)
    save_classification_report(report, OUTPUT_METRICS_TABLE)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Packet Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(OUTPUT_CONFUSION_PLOT)
    plt.close()
    print(f"[INFO] Packet confusion matrix saved to {OUTPUT_CONFUSION_PLOT}")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.6f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Packet ROC Curve")
    plt.legend()
    plt.savefig(OUTPUT_ROC_PLOT)
    plt.close()
    print(f"[INFO] Packet ROC curve saved to {OUTPUT_ROC_PLOT}")

    save_metrics(
        {
            "roc_auc": float(roc_auc),
            "test_rows": X_test.shape[0],
            "feature_count": X_test.shape[1],
        },
        OUTPUT_METRICS_PATH,
    )


if __name__ == "__main__":
    main()
