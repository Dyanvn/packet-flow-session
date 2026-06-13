import os
import time
import platform
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False


DATASET_PATH = "data/processed/packet_model_dataset.csv"
OUTPUT_PATH = "outputs/metrics/packet_performance_report.csv"


def ensure_directories():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


def load_dataset(path=DATASET_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found. Run src/pipeline/build_packet_dataset.py first or create {path}."
        )
    df = pd.read_csv(path)
    if "Label" not in df.columns:
        raise KeyError("Label column not found in packet dataset")
    # Map protocol to numeric code and select numeric features used in training
    PROTOCOL_MAP = {"TCP": 1, "UDP": 0}
    if "Protocol" in df.columns:
        df["Protocol Code"] = df["Protocol"].map(PROTOCOL_MAP).fillna(2).astype(int)
    else:
        df["Protocol Code"] = 2

    selected = [
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

    missing = [c for c in selected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing packet features for benchmark: {missing}")

    X = df[selected]
    y = df["Label"].astype(int)
    return X, y


def measure_memory_cpu():
    if not PSUTIL_AVAILABLE:
        return None
    proc = psutil.Process(os.getpid())
    mem = proc.memory_info().rss
    cpu = proc.cpu_percent(interval=0.1)
    return {"memory_rss": mem, "cpu_percent": cpu}


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

    metrics = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "train_rows": X_train.shape[0],
        "test_rows": X_test.shape[0],
        "feature_count": X_train.shape[1],
    }

    pre_stats = measure_memory_cpu()
    if pre_stats:
        metrics.update({f"before_{k}": v for k, v in pre_stats.items()})

    model_path = "outputs/models/xgboost_packet_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Packet model not found at {model_path}. Train it first.")

    model = joblib.load(model_path)

    start = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_time = time.perf_counter() - start
    throughput = X_test.shape[0] / inference_time if inference_time > 0 else 0.0

    metrics.update(
        {
            "inference_time_seconds": round(inference_time, 6),
            "throughput_samples_per_second": round(throughput, 2),
            "avg_latency_milliseconds": round(inference_time / X_test.shape[0] * 1000, 6),
        }
    )

    post_stats = measure_memory_cpu()
    if post_stats:
        metrics.update({f"after_{k}": v for k, v in post_stats.items()})

    df = pd.DataFrame([metrics])
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[INFO] Packet benchmark saved to {OUTPUT_PATH}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
