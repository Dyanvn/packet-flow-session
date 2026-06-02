import os
import time
import math
import joblib
import pandas as pd

from concurrent.futures import ThreadPoolExecutor

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False


def measure_memory_cpu():
    if not PSUTIL_AVAILABLE:
        return None
    proc = psutil.Process(os.getpid())
    mem = proc.memory_info().rss
    cpu = proc.cpu_percent(interval=0.1)
    return {"memory_rss": mem, "cpu_percent": cpu}


def chunkify(df, n_chunks):
    n = len(df)
    if n == 0:
        return []
    chunk_size = math.ceil(n / n_chunks)
    return [df[i:i+chunk_size] for i in range(0, n, chunk_size)]


def run_parallel_inference(model_path, X_test, workers=4):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)

    chunks = chunkify(X_test, workers)
    pre = measure_memory_cpu()
    start = time.perf_counter()

    def predict_chunk(df_chunk):
        # Ensure numpy array input
        return model.predict(df_chunk)

    with ThreadPoolExecutor(max_workers=workers) as exe:
        futures = [exe.submit(predict_chunk, c) for c in chunks]
        results = [f.result() for f in futures]

    total_time = time.perf_counter() - start
    post = measure_memory_cpu()

    total_samples = len(X_test)
    throughput = total_samples / total_time if total_time > 0 else 0.0
    avg_latency_ms = (total_time / total_samples) * 1000 if total_samples > 0 else 0.0

    metrics = {
        "workers": workers,
        "inference_time_seconds": round(total_time, 6),
        "throughput_samples_per_second": round(throughput, 2),
        "avg_latency_milliseconds": round(avg_latency_ms, 6),
    }
    if pre:
        metrics.update({f"before_{k}": v for k, v in pre.items()})
    if post:
        metrics.update({f"after_{k}": v for k, v in post.items()})

    return metrics
