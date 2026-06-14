# src/evaluation/demo_live_attack_pcap.py

from pathlib import Path
import pandas as pd

from src.models.ensemble.runtime_inference import (
    load_packet_model,
    load_flow_model,
    load_session_model,
    infer_packet_batch,
    infer_flow_batch,
    infer_session_batch,
)
from src.models.ensemble.ensemble_inference import simple_ensemble_rule


def demo_live_attack_pcap(
    packet_csv: str | Path = "data/processed/packet_attack_dataset.csv",
    flow_csv: str | Path = "data/processed/flow_attack_dataset.csv",
    session_csv: str | Path = "data/processed/session_attack_dataset.csv",
    num_samples: int = 10,
):
    packet_csv = Path(packet_csv)
    flow_csv = Path(flow_csv)
    session_csv = Path(session_csv)

    if not (packet_csv.exists() and flow_csv.exists() and session_csv.exists()):
        raise FileNotFoundError(
            f"Thiếu dataset attack. Hãy chạy: python -m src.pipeline.run_attack_pcap_pipeline trước."
        )

    # Đọc dataset attack
    df_packet = pd.read_csv(packet_csv)
    df_flow = pd.read_csv(flow_csv)
    df_session = pd.read_csv(session_csv)

    print(f"[LiveAttackDemo] Packet attack dataset : {df_packet.shape}")
    print(f"[LiveAttackDemo] Flow attack dataset   : {df_flow.shape}")
    print(f"[LiveAttackDemo] Session attack dataset: {df_session.shape}")

    # Cắt cho align số dòng (demo đơn giản)
    n = min(len(df_packet), len(df_flow), len(df_session))
    df_packet = df_packet.head(n).reset_index(drop=True)
    df_flow = df_flow.head(n).reset_index(drop=True)
    df_session = df_session.head(n).reset_index(drop=True)

    # Load models
    packet_model = load_packet_model()
    flow_model = load_flow_model()
    session_model = load_session_model()

    # Inference từng tầng
    df_packet_out = infer_packet_batch(df_packet, packet_model)
    df_flow_out = infer_flow_batch(df_flow, flow_model)
    df_session_out = infer_session_batch(df_session, session_model)

    # Tính ensemble cho từng record
    records = []
    for i in range(n):
        p_pred = int(df_packet_out.loc[i, "packet_pred"])
        p_proba = float(df_packet_out.loc[i, "packet_proba"])

        f_pred = int(df_flow_out.loc[i, "flow_pred"])
        f_proba = float(df_flow_out.loc[i, "flow_proba"])

        s_pred = int(df_session_out.loc[i, "session_pred"])
        s_proba = float(df_session_out.loc[i, "session_proba"])

        ens = simple_ensemble_rule(
            packet_pred=p_pred,
            packet_proba=p_proba,
            flow_pred=f_pred,
            flow_proba=f_proba,
            session_pred=s_pred,
            session_proba=s_proba,
        )

        records.append(
            {
                "packet_pred": p_pred,
                "packet_proba": p_proba,
                "flow_pred": f_pred,
                "flow_proba": f_proba,
                "session_pred": s_pred,
                "session_proba": s_proba,
                "ensemble_pred": ens,
            }
        )

    df_result = pd.DataFrame(records)

    # Chọn top các record có xác suất cao nhất ở flow hoặc session (nghi ngờ tấn công)
    df_result["max_proba"] = df_result[["flow_proba", "session_proba", "packet_proba"]].max(axis=1)
    df_top = df_result.sort_values("max_proba", ascending=False).head(num_samples)

    print("\n=== DEMO LIVE ATTACK FROM PCAP ===")
    for idx, row in df_top.iterrows():
        print("\n--- Record index", idx, "---")
        print(f"Packet model  : pred={row['packet_pred']}, proba={row['packet_proba']:.3f}")
        print(f"Flow model    : pred={row['flow_pred']}, proba={row['flow_proba']:.3f}")
        print(f"Session model : pred={row['session_pred']}, proba={row['session_proba']:.3f}")
        print(f"Ensemble      : pred={row['ensemble_pred']}, max_proba={row['max_proba']:.3f}")


if __name__ == "__main__":
    demo_live_attack_pcap(num_samples=10)