# src/features/session/session_features.py

import pandas as pd
from pathlib import Path

def load_flow_dataset(flow_csv_path: str | Path) -> pd.DataFrame:
    """
    Load flow_model_dataset.csv

    Expected columns:
    - Flow Duration
    - Total Fwd Packets
    - Total Backward Packets
    - SYN Flag Count
    - ACK Flag Count
    - Average Packet Size
    - Packet Length Mean
    - Label
    """

    flow_csv_path = Path(flow_csv_path)
    df = pd.read_csv(flow_csv_path)

    return df

def assign_time_window(
    df: pd.DataFrame,
    window_size_sec: int = 100
) -> pd.DataFrame:

    df = df.copy()

    df["window_index"] = (
        df.index // window_size_sec
    ).astype(int)

    return df

def build_session_features(df_flows: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo Session Dataset từ Flow Dataset.

    Mỗi session được xác định bởi window_index
    (được tạo ở hàm assign_time_window).
    """

    group_cols = ["window_index"]

    agg_dict = {
        "Flow Duration": ["mean", "max"],
        "Total Fwd Packets": ["sum", "mean"],
        "Total Backward Packets": ["sum", "mean"],
        "Average Packet Size": ["mean"],
        "Packet Length Mean": ["mean"],
        "Label": ["sum", "count"]
    }

    grouped = df_flows.groupby(group_cols).agg(agg_dict)

    grouped.columns = [
        "_".join(col).strip()
        for col in grouped.columns.to_flat_index()
    ]

    grouped = grouped.reset_index()

    grouped.rename(
        columns={
            "Flow Duration_mean": "avg_flow_duration",
            "Flow Duration_max": "max_flow_duration",

            "Total Fwd Packets_sum": "total_fwd_packets",
            "Total Fwd Packets_mean": "avg_fwd_packets",

            "Total Backward Packets_sum": "total_bwd_packets",
            "Total Backward Packets_mean": "avg_bwd_packets",

            "Average Packet Size_mean": "avg_packet_size",
            "Packet Length Mean_mean": "avg_packet_length",

            "Label_sum": "num_attack_flows",
            "Label_count": "num_flows",
        },
        inplace=True,
    )

    grouped["session_id"] = (
        "session_"
        + grouped["window_index"].astype(str)
    )

    return grouped

def build_session_dataset(
    flow_csv_path: str | Path,
    output_csv_path: str | Path,
    window_size_sec: int = 100,
) -> pd.DataFrame:
    df_flows = load_flow_dataset(flow_csv_path)
    df_flows = assign_time_window(df_flows, window_size_sec=window_size_sec)
    df_sessions = build_session_features(df_flows)
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_sessions.to_csv(output_csv_path, index=False)
    return df_sessions

if __name__ == "__main__":
    flow_csv = "data/processed/flow_model_dataset.csv"
    session_csv = "data/processed/session_model_dataset.csv"
    build_session_dataset(flow_csv, session_csv, window_size_sec=100)