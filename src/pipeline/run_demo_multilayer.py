# src/pipeline/run_demo_multilayer.py

from pathlib import Path
import pandas as pd

import src.pipeline.run_session_pipeline as rsp
from src.pipeline.run_session_pipeline import run_session_pipeline
from src.workers.pipeline_workers import run_parallel_demo


def ensure_flow_dataset():
    flow_csv = Path("data/processed/flow_model_dataset.csv")
    if not flow_csv.exists():
        print("[Demo] flow_model_dataset.csv chưa có, chạy flow pipeline...")
        run_flow_pipeline()
    else:
        print("[Demo] flow_model_dataset.csv đã tồn tại.")


def ensure_session_dataset():
    session_csv = Path("data/processed/session_model_dataset.csv")
    if not session_csv.exists():
        print("[Demo] session_model_dataset.csv chưa có, chạy session pipeline...")
        rsp.run_session_pipeline()
    else:
        print("[Demo] session_model_dataset.csv đã tồn tại.")

def ensure_models():
    required_models = [
        "outputs/models/xgboost_packet_model.pkl",
        "outputs/models/xgboost_flow_model.pkl",
        "outputs/models/xgboost_session_model.pkl",
    ]

    for model_path in required_models:
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"[Demo] Thiếu model: {model_path}"
            )
        

def print_dataset_stats():

    packet_df = pd.read_csv(
        "data/processed/packet_model_dataset.csv"
    )

    flow_df = pd.read_csv(
        "data/processed/flow_model_dataset.csv"
    )

    session_df = pd.read_csv(
        "data/processed/session_model_dataset.csv"
    )

    print(
        f"[Demo] Packet Dataset : {packet_df.shape}"
    )

    print(
        f"[Demo] Flow Dataset   : {flow_df.shape}"
    )

    print(
        f"[Demo] Session Dataset: {session_df.shape}"
    )

def main():
    print("=== DEMO: Multi-layer + Parallel Inference ===")

    ensure_flow_dataset()
    ensure_session_dataset()

    print_dataset_stats()

    ensure_models()

    print("[Demo] Chạy parallel pipeline demo...")
    run_parallel_demo(batch_size=500)

    print("=== DEMO DONE ===")


if __name__ == "__main__":
    main()