# src/evaluation/demo_realtime_console.py

from pathlib import Path
import time
import pandas as pd
import csv

from datetime import datetime

from src.models.ensemble.runtime_inference import (
    load_packet_model,
    load_flow_model,
    load_session_model,
    infer_packet_batch,
    infer_flow_batch,
    infer_session_batch,
)
from src.models.ensemble.ensemble_inference import simple_ensemble_rule


def demo_realtime_console(
    packet_csv: str | Path = "data/processed/packet_attack_dataset.csv",
    flow_csv: str | Path = "data/processed/flow_attack_dataset.csv",
    session_csv: str | Path = "data/processed/session_attack_dataset.csv",
    batch_size: int = 1000,
    sleep_sec: float = 0.5,
):
    packet_csv = Path(packet_csv)
    flow_csv = Path(flow_csv)
    session_csv = Path(session_csv)

    if not (packet_csv.exists() and flow_csv.exists() and session_csv.exists()):
        raise FileNotFoundError(
            "Thiếu dataset attack. Hãy chạy: python -m src.pipeline.run_attack_pcap_pipeline trước."
        )

    # =========================
    # Đọc dataset
    # =========================
    df_packet = pd.read_csv(packet_csv)
    df_flow = pd.read_csv(flow_csv)
    df_session = pd.read_csv(session_csv)

    print(f"[RTDemo] Packet dataset : {df_packet.shape}")
    print(f"[RTDemo] Flow dataset   : {df_flow.shape}")
    print(f"[RTDemo] Session dataset: {df_session.shape}")

    # Debug tên cột (có thể comment sau)
    print("\n[RTDemo] Packet Columns:")
    print(df_packet.columns.tolist())

    # Align số record giữa 3 tầng
    n = min(len(df_packet), len(df_flow), len(df_session))

    df_packet = df_packet.head(n).reset_index(drop=True)
    df_flow = df_flow.head(n).reset_index(drop=True)
    df_session = df_session.head(n).reset_index(drop=True)

    # =========================
    # Load models
    # =========================
    packet_model = load_packet_model()
    flow_model = load_flow_model()
    session_model = load_session_model()

    print(f"\n[RTDemo] Start streaming {n} records")
    print(f"[RTDemo] Batch size = {batch_size}")

    # =========================
    # Log file
    # =========================
    LOG_PATH = Path("outputs/metrics/realtime_alerts_log.csv")

    LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    first_write = not LOG_PATH.exists()

    with LOG_PATH.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as log_file:

        writer = csv.DictWriter(
            log_file,
            fieldnames=[
                "log_time",
                "packet_time",
                "src_ip",
                "src_port",
                "dst_ip",
                "dst_port",
                "protocol",
                "packet_proba",
                "flow_proba",
                "session_proba",
                "ensemble_pred",
            ],
        )

        if first_write:
            writer.writeheader()

        # =========================
        # Streaming simulation
        # =========================
        for batch_idx, start in enumerate(
            range(0, n, batch_size),
            start=1
        ):
            end = min(start + batch_size, n)

            batch_packet = (
                df_packet.iloc[start:end]
                .reset_index(drop=True)
            )

            batch_flow = (
                df_flow.iloc[start:end]
                .reset_index(drop=True)
            )

            batch_session = (
                df_session.iloc[start:end]
                .reset_index(drop=True)
            )

            # =========================
            # Inference
            # =========================
            out_packet = infer_packet_batch(
                batch_packet,
                packet_model
            )

            out_flow = infer_flow_batch(
                batch_flow,
                flow_model
            )

            out_session = infer_session_batch(
                batch_session,
                session_model
            )

            alerts = 0
            alert_samples = []

            # =========================
            # Ensemble
            # =========================
            for i in range(len(batch_packet)):

                packet_pred = int(
                    out_packet.loc[i, "packet_pred"]
                )

                packet_proba = float(
                    out_packet.loc[i, "packet_proba"]
                )

                flow_pred = int(
                    out_flow.loc[i, "flow_pred"]
                )

                flow_proba = float(
                    out_flow.loc[i, "flow_proba"]
                )

                session_pred = int(
                    out_session.loc[i, "session_pred"]
                )

                session_proba = float(
                    out_session.loc[i, "session_proba"]
                )

                ens = simple_ensemble_rule(
                    packet_pred=packet_pred,
                    packet_proba=packet_proba,
                    flow_pred=flow_pred,
                    flow_proba=flow_proba,
                    session_pred=session_pred,
                    session_proba=session_proba,
                )

                # =========================
                # Alert
                # =========================
                if ens == 1:

                    alerts += 1

                    if len(alert_samples) < 3:
                        alert_samples.append(
                            {
                                "packet_proba": packet_proba,
                                "flow_proba": flow_proba,
                                "session_proba": session_proba,
                            }
                        )

                    writer.writerow(
                        {
                            "log_time": datetime.now().isoformat(),

                            "packet_time":
                            batch_packet.loc[i, "Timestamp"]
                            if "Timestamp" in batch_packet.columns
                            else "",

                            "src_ip":
                            batch_packet.loc[i, "Src IP"]
                            if "Src IP" in batch_packet.columns
                            else "",

                            "src_port":
                            batch_packet.loc[i, "Src Port"]
                            if "Src Port" in batch_packet.columns
                            else "",

                            "dst_ip":
                            batch_packet.loc[i, "Dst IP"]
                            if "Dst IP" in batch_packet.columns
                            else "",

                            "dst_port":
                            batch_packet.loc[i, "Dst Port"]
                            if "Dst Port" in batch_packet.columns
                            else "",

                            "protocol":
                            batch_packet.loc[i, "Protocol"]
                            if "Protocol" in batch_packet.columns
                            else "",

                            "packet_proba": packet_proba,
                            "flow_proba": flow_proba,
                            "session_proba": session_proba,
                            "ensemble_pred": 1,
                        }
                    )

            log_file.flush()

            print(
                f"[RTDemo] Batch {batch_idx} "
                f"({start}:{end}) -> "
                f"{alerts} alerts"
            )

            for j, a in enumerate(
                alert_samples,
                start=1
            ):
                print(
                    f"    Alert {j}: "
                    f"packet_proba={a['packet_proba']:.3f}, "
                    f"flow_proba={a['flow_proba']:.3f}, "
                    f"session_proba={a['session_proba']:.3f}"
                )

            time.sleep(sleep_sec)

    print(
        f"\n[RTDemo] Log saved to: {LOG_PATH}"
    )


if __name__ == "__main__":
    demo_realtime_console()