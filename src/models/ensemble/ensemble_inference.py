from pathlib import Path
import joblib
import pandas as pd
from xgboost import XGBClassifier  # chỉ để type hint (không bắt buộc)


def load_models(
    packet_model_path: str | Path = "outputs/models/xgboost_packet_model.pkl",
    flow_model_path: str | Path = "outputs/models/xgboost_flow_model.pkl",
    session_model_path: str | Path = "outputs/models/xgboost_session_model.pkl",
):
    packet_model = joblib.load(packet_model_path)
    flow_model = joblib.load(flow_model_path)
    session_model = joblib.load(session_model_path)
    return packet_model, flow_model, session_model


def load_datasets(
    packet_csv: str | Path = "data/processed/packet_model_dataset.csv",
    flow_csv: str | Path = "data/processed/flow_model_dataset.csv",
    session_csv: str | Path = "data/processed/session_model_dataset.csv",
):
    df_packet = pd.read_csv(packet_csv)
    df_flow = pd.read_csv(flow_csv)
    df_session = pd.read_csv(session_csv)
    return df_packet, df_flow, df_session


def simple_ensemble_rule(
    packet_pred: int,
    packet_proba: float,
    flow_pred: int,
    flow_proba: float,
    session_pred: int,
    session_proba: float,
    high_th: float = 0.8,
) -> int:
    """
    Rule-based ensemble đơn giản:
    - Nếu packet hoặc flow có proba > high_th và pred=1 → trả về 1 (attack mạnh).
    - Ngược lại nếu session_pred=1 → trả về 1 (attack theo context).
    - Còn lại → 0.
    """
    if packet_pred == 1 and packet_proba >= high_th:
        return 1
    if flow_pred == 1 and flow_proba >= high_th:
        return 1
    if session_pred == 1:
        return 1
    return 0


def run_offline_ensemble_demo():
    """
    Demo offline:
    - Lấy một subset nhỏ packet/flow/session
    - Cho model dự đoán proba
    - Áp rule ensemble
    - Lưu kết quả ra CSV để xem.
    """
    packet_model, flow_model, session_model = load_models()
    df_packet, df_flow, df_session = load_datasets()

    # =========================
    # Tạo Protocol Code giống lúc train packet model
    # =========================
    PROTOCOL_MAP = {"TCP": 1, "UDP": 0}

    if "Protocol" in df_packet.columns:
        df_packet["Protocol Code"] = (
            df_packet["Protocol"]
            .map(PROTOCOL_MAP)
            .fillna(2)
            .astype(int)
        )

    # =========================
    # Align số lượng record
    # =========================
    n = min(len(df_packet), len(df_flow), len(df_session))

    df_packet = df_packet.head(n)
    df_flow = df_flow.head(n)
    df_session = df_session.head(n)

    # =========================
    # Packet features
    # Phải giống hệt lúc train
    # =========================
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

    X_packet = df_packet[packet_features]

    # =========================
    # Flow features
    # =========================
    X_flow = df_flow.drop(columns=["Label"])

    # =========================
    # Session features
    # Phải giống hệt lúc train session model
    # =========================
    X_session = df_session[
        [
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
    ]

    # =========================
    # Predict
    # =========================
    packet_proba = packet_model.predict_proba(X_packet)[:, 1]
    packet_pred = packet_model.predict(X_packet)

    flow_proba = flow_model.predict_proba(X_flow)[:, 1]
    flow_pred = flow_model.predict(X_flow)

    session_proba = session_model.predict_proba(X_session)[:, 1]
    session_pred = session_model.predict(X_session)

    # =========================
    # Ensemble Rule
    # =========================
    ensemble_preds = []

    for i in range(n):
        final_y = simple_ensemble_rule(
            packet_pred=int(packet_pred[i]),
            packet_proba=float(packet_proba[i]),
            flow_pred=int(flow_pred[i]),
            flow_proba=float(flow_proba[i]),
            session_pred=int(session_pred[i]),
            session_proba=float(session_proba[i]),
        )

        ensemble_preds.append(final_y)

    # =========================
    # Save results
    # =========================
    df_result = pd.DataFrame(
        {
            "packet_pred": packet_pred,
            "packet_proba": packet_proba,
            "flow_pred": flow_pred,
            "flow_proba": flow_proba,
            "session_pred": session_pred,
            "session_proba": session_proba,
            "ensemble_pred": ensemble_preds,
        }
    )

    output_path = Path("outputs/metrics/ensemble_demo_results.csv")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_result.to_csv(output_path, index=False)

    print(f"[Ensemble] Offline demo results saved to: {output_path}")

def ensemble_batch(
    df_packet_out: pd.DataFrame,
    df_flow_out: pd.DataFrame,
    df_session_out: pd.DataFrame,
) -> pd.DataFrame:
    """
    Nhận 3 DataFrame đã có *_pred, *_proba (cùng số dòng, align theo index),
    trả về DataFrame mới có thêm cột ensemble_pred.
    """
    n = min(len(df_packet_out), len(df_flow_out), len(df_session_out))

    df_packet_out = df_packet_out.head(n).reset_index(drop=True)
    df_flow_out = df_flow_out.head(n).reset_index(drop=True)
    df_session_out = df_session_out.head(n).reset_index(drop=True)

    ensemble_preds = []
    for i in range(n):
        final_y = simple_ensemble_rule(
            packet_pred=int(df_packet_out.loc[i, "packet_pred"]),
            packet_proba=float(df_packet_out.loc[i, "packet_proba"]),
            flow_pred=int(df_flow_out.loc[i, "flow_pred"]),
            flow_proba=float(df_flow_out.loc[i, "flow_proba"]),
            session_pred=int(df_session_out.loc[i, "session_pred"]),
            session_proba=float(df_session_out.loc[i, "session_proba"]),
        )
        ensemble_preds.append(final_y)

    df_ens = pd.DataFrame(
        {
            "packet_pred": df_packet_out["packet_pred"],
            "packet_proba": df_packet_out["packet_proba"],
            "flow_pred": df_flow_out["flow_pred"],
            "flow_proba": df_flow_out["flow_proba"],
            "session_pred": df_session_out["session_pred"],
            "session_proba": df_session_out["session_proba"],
            "ensemble_pred": ensemble_preds,
        }
    )

    return df_ens

if __name__ == "__main__":
    run_offline_ensemble_demo()