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

from src.preprocessing.feature_selection import selected_features


DATASET_PATH = "data/processed/flow_model_dataset.csv"
OUTPUT_METRICS_PATH = "outputs/metrics/flow_evaluation_metrics.csv"
OUTPUT_CONFUSION_PLOT = "outputs/figures/confusion_matrix.png"
OUTPUT_ROC_PLOT = "outputs/figures/roc_curve.png"
OUTPUT_METRICS_TABLE = "outputs/metrics/flow_classification_report.csv"


def ensure_directories():
    os.makedirs(os.path.dirname(OUTPUT_METRICS_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_CONFUSION_PLOT), exist_ok=True)


def load_dataset(path=DATASET_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found. Run src/pipeline/build_flow_dataset.py first or create {path}."
        )
    df = pd.read_csv(path)
    X = df[selected_features]
    y = df["Label"]
    return X, y


def save_classification_report(report, path):
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(path, index=True)
    print(f"[INFO] Classification report saved to {path}")


def save_metrics(metrics, path):
    df = pd.DataFrame([metrics])
    df.to_csv(path, index=False)
    print(f"[INFO] Evaluation metrics saved to {path}")


def main():
    ensure_directories()

    X, y = load_dataset()
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = joblib.load("outputs/models/xgboost_flow_model.pkl")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, digits=6, output_dict=True)
    print(classification_report(y_test, y_pred, digits=6))
    save_classification_report(report, OUTPUT_METRICS_TABLE)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(OUTPUT_CONFUSION_PLOT)
    plt.close()
    print(f"[INFO] Confusion matrix saved to {OUTPUT_CONFUSION_PLOT}")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.6f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(OUTPUT_ROC_PLOT)
    plt.close()
    print(f"[INFO] ROC curve saved to {OUTPUT_ROC_PLOT}")

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
