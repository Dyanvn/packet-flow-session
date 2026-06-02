import os
import time
import platform
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from src.preprocessing.feature_selection import selected_features

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False


DATASET_PATH = "data/processed/flow_model_dataset.csv"
OUTPUT_PATH = "outputs/metrics/flow_performance_report.csv"


def ensure_directories():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


def load_dataset(path=DATASET_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found. Run src/pipeline/build_flow_dataset.py first or create {path}."
        )
    df = pd.read_csv(path)
    X = df[selected_features]
    y = df["Label"]
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

    model = joblib.load("outputs/models/xgboost_flow_model.pkl")

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
    print(f"[INFO] Benchmark saved to {OUTPUT_PATH}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
