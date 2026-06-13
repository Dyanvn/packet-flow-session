# src/workers/pipeline_workers.py

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import pandas as pd


from src.models.ensemble.runtime_inference import (
    load_packet_model,
    load_flow_model,
    load_session_model,
    infer_packet_batch,
    infer_flow_batch,
    infer_session_batch,
)
from src.models.ensemble.ensemble_inference import ensemble_batch


def packet_stage(df_packet: pd.DataFrame):
    model = load_packet_model()
    return infer_packet_batch(df_packet, model)


def flow_stage(df_flow: pd.DataFrame):
    model = load_flow_model()
    return infer_flow_batch(df_flow, model)


def session_stage(df_session: pd.DataFrame):
    model = load_session_model()
    return infer_session_batch(df_session, model)


def run_parallel_demo(batch_size: int = 1000):
    import pandas as pd
    import time
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    # =========================
    # LOAD DATA
    # =========================
    df_packet = pd.read_csv("data/processed/packet_model_dataset.csv")
    df_flow = pd.read_csv("data/processed/flow_model_dataset.csv")
    df_session = pd.read_csv("data/processed/session_model_dataset.csv")

    n = min(len(df_packet), len(df_flow), len(df_session))

    df_packet = df_packet.head(n)
    df_flow = df_flow.head(n)
    df_session = df_session.head(n)

    batches = range(0, n, batch_size)

    results_packet = []
    results_flow = []
    results_session = []

    # =========================
    # START TIMER
    # =========================
    t_start = time.perf_counter()

    # =========================
    # PARALLEL EXECUTION
    # =========================
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []

        for start in batches:
            end = start + batch_size

            futures.append(
                (
                    executor.submit(packet_stage, df_packet.iloc[start:end]),
                    executor.submit(flow_stage, df_flow.iloc[start:end]),
                    executor.submit(session_stage, df_session.iloc[start:end]),
                )
            )

        for fut_packet, fut_flow, fut_session in futures:
            results_packet.append(fut_packet)
            results_flow.append(fut_flow)
            results_session.append(fut_session)

    # =========================
    # COLLECT RESULTS
    # =========================
    packet_out = pd.concat([f.result() for f in results_packet], ignore_index=True)
    flow_out = pd.concat([f.result() for f in results_flow], ignore_index=True)
    session_out = pd.concat([f.result() for f in results_session], ignore_index=True)

    # =========================
    # ENSEMBLE INFERENCE
    # =========================
    ensemble_out = ensemble_batch(packet_out, flow_out, session_out)

    # =========================
    # END TIMER
    # =========================
    t_end = time.perf_counter()

    total_time = t_end - t_start
    throughput = n / total_time if total_time > 0 else 0

    # =========================
    # PRINT RESULTS
    # =========================
    print("[Workers] Packet out:", packet_out.shape)
    print("[Workers] Flow out:", flow_out.shape)
    print("[Workers] Session out:", session_out.shape)
    print("[Workers] Ensemble out:", ensemble_out.shape)

    print(f"[Workers] Total time: {total_time:.4f} s")
    print(f"[Workers] Throughput: {throughput:.2f} records/s")

    # =========================
    # SAVE OUTPUTS
    # =========================
    output_dir = Path("outputs/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    packet_out.head(100).to_csv(output_dir / "parallel_packet_sample.csv", index=False)
    flow_out.head(100).to_csv(output_dir / "parallel_flow_sample.csv", index=False)
    session_out.head(100).to_csv(output_dir / "parallel_session_sample.csv", index=False)
    ensemble_out.head(100).to_csv(output_dir / "parallel_ensemble_sample.csv", index=False)

    # =========================
    # BENCHMARK SAVE
    # =========================
    benchmark_df = pd.DataFrame([{
        "num_records": n,
        "batch_size": batch_size,
        "total_time_sec": total_time,
        "throughput_records_per_sec": throughput
    }])

    benchmark_path = output_dir / "parallel_pipeline_performance.csv"
    benchmark_df.to_csv(benchmark_path, index=False)

    print(f"[Workers] Benchmark saved to: {benchmark_path}")


if __name__ == "__main__":
    run_parallel_demo(batch_size=500)