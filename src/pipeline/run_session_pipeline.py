# src/pipeline/run_session_pipeline.py

from pathlib import Path

from src.features.session.session_features import build_session_dataset

def run_session_pipeline(
    flow_csv_path: str | Path = "data/processed/flow_model_dataset.csv",
    session_csv_path: str | Path = "data/processed/session_model_dataset.csv",
    window_size_sec: int = 100,
) -> None:
    flow_csv_path = Path(flow_csv_path)
    session_csv_path = Path(session_csv_path)

    if not flow_csv_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {flow_csv_path}. "
            "Hãy chạy flow pipeline trước: python -m src.pipeline.run_flow_pipeline"
        )

    print(f"[SessionPipeline] Đọc flow dataset từ: {flow_csv_path}")
    df_sessions = build_session_dataset(
        flow_csv_path=flow_csv_path,
        output_csv_path=session_csv_path,
        window_size_sec=window_size_sec,
    )

    print(f"[SessionPipeline] Đã tạo session dataset: {session_csv_path}")
    print(f"[SessionPipeline] Số session: {len(df_sessions)}")

if __name__ == "__main__":
    run_session_pipeline()