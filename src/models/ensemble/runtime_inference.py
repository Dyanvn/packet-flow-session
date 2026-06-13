# src/models/ensemble/runtime_inference.py

from pathlib import Path
from src.features.flow import flow_features
import joblib
import pandas as pd


# ====== Load models dùng lại nhiều nơi ======

def load_packet_model(
    model_path: str | Path = "outputs/models/xgboost_packet_model.pkl",
):
    return joblib.load(model_path)


def load_flow_model(
    model_path: str | Path = "outputs/models/xgboost_flow_model.pkl",
):
    return joblib.load(model_path)


def load_session_model(
    model_path: str | Path = "outputs/models/xgboost_session_model.pkl",
):
    return joblib.load(model_path)


# ====== Inference helpers cho từng tầng ======

def infer_packet_batch(df_packet: pd.DataFrame, model) -> pd.DataFrame:
    """
    Nhận DataFrame packet đã có đủ feature, trả về DataFrame có thêm:
    - packet_pred
    - packet_proba
    """
    PROTOCOL_MAP = {"TCP": 1, "UDP": 0}

    if "Protocol" in df_packet.columns and "Protocol Code" not in df_packet.columns:
        df_packet = df_packet.copy()
        df_packet["Protocol Code"] = (
            df_packet["Protocol"]
            .map(PROTOCOL_MAP)
            .fillna(2)
            .astype(int)
        )

    packet_features = [
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

    X = df_packet[packet_features]
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)

    df_out = df_packet.copy()
    df_out["packet_pred"] = pred
    df_out["packet_proba"] = proba
    return df_out


def infer_flow_batch(df_flow: pd.DataFrame, model) -> pd.DataFrame:
    flow_features = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "SYN Flag Count",
    "ACK Flag Count",
    "Average Packet Size",
    "Packet Length Mean",
]

    X = df_flow[flow_features]
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)

    df_out = df_flow.copy()
    df_out["flow_pred"] = pred
    df_out["flow_proba"] = proba
    return df_out


def infer_session_batch(df_session: pd.DataFrame, model) -> pd.DataFrame:
    """
    Nhận DataFrame session_model_dataset (hoặc tương đương),
    trả về thêm:
    - session_pred
    - session_proba
    """
    feature_cols = [
        "avg_flow_duration",
        "max_flow_duration",
        "total_fwd_packets",
        "avg_fwd_packets",
        "total_bwd_packets",
        "avg_bwd_packets",
        "avg_packet_size",
        "avg_packet_length",
        "num_flows",
    ]

    X = df_session[feature_cols]
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)

    df_out = df_session.copy()
    df_out["session_pred"] = pred
    df_out["session_proba"] = proba
    return df_out


if __name__ == "__main__":
    # Demo nhỏ: đọc dataset đã có và chạy thử 1 batch
    packet_model = load_packet_model()
    flow_model = load_flow_model()
    session_model = load_session_model()

    df_packet = pd.read_csv("data/processed/packet_model_dataset.csv").head(100)
    df_flow = pd.read_csv("data/processed/flow_model_dataset.csv").head(100)
    df_session = pd.read_csv("data/processed/session_model_dataset.csv").head(100)

    df_packet_out = infer_packet_batch(df_packet, packet_model)
    df_flow_out = infer_flow_batch(df_flow, flow_model)
    df_session_out = infer_session_batch(df_session, session_model)

    print("[RuntimeInference] Demo packet batch:", df_packet_out.shape)
    print("[RuntimeInference] Demo flow batch:", df_flow_out.shape)
    print("[RuntimeInference] Demo session batch:", df_session_out.shape)