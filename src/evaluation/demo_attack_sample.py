# src/evaluation/demo_attack_sample.py

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


def demo_attack_sample(
    packet_csv: str | Path = "data/processed/packet_model_dataset.csv",
    flow_csv: str | Path = "data/processed/flow_model_dataset.csv",
    session_csv: str | Path = "data/processed/session_model_dataset.csv",
    num_samples: int = 100,
):
    packet_csv = Path(packet_csv)
    flow_csv = Path(flow_csv)
    session_csv = Path(session_csv)

    df_packet = pd.read_csv(packet_csv)
    df_flow = pd.read_csv(flow_csv)
    df_session = pd.read_csv(session_csv)

    # Lấy một số flow có Label = 1 (attack)
    df_attack_flow = df_flow[df_flow["Label"] == 1].head(num_samples)
    if df_attack_flow.empty:
        print("[DemoAttack] Không tìm thấy flow tấn công (Label=1) trong dataset.")
        return

    # Giả sử index tương đối gần nhau giữa packet/flow/session (demo offline)
    indices = df_attack_flow.index.tolist()

    df_packet_sample = df_packet.iloc[indices].reset_index(drop=True)
    df_flow_sample = df_attack_flow.reset_index(drop=True)
    df_session_sample = df_session.head(len(indices)).reset_index(drop=True)

    # Load models
    packet_model = load_packet_model()
    flow_model = load_flow_model()
    session_model = load_session_model()

    # Inference từng tầng
    df_packet_out = infer_packet_batch(df_packet_sample, packet_model)
    df_flow_out = infer_flow_batch(df_flow_sample, flow_model)
    df_session_out = infer_session_batch(df_session_sample, session_model)

    print("=== DEMO ATTACK SAMPLES (FLOW Label = 1) ===")
    for i in range(len(indices)):
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

        print(f"\n--- Sample {i} (flow Label=1) ---")
        print(f"Packet model  : pred={p_pred}, proba={p_proba:.3f}")
        print(f"Flow model    : pred={f_pred}, proba={f_proba:.3f}")
        print(f"Session model : pred={s_pred}, proba={s_proba:.3f}")
        print(f"Ensemble      : pred={ens}")

if __name__ == "__main__":
    demo_attack_sample(num_samples=100)