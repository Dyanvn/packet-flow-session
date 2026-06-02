import os

import pandas as pd

from src.ingestion.packet_reader import load_packets
from src.features.packet.packet_features import extract_packet_features
from src.features.flow.flow_builder import build_flows
from src.features.flow.flow_features import extract_flow_features


OUTPUT_DIR = "outputs/features"


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    ensure_output_dir(OUTPUT_DIR)

    packets = load_packets(
        "data/sample/Friday-WorkingHours.pcap",
        limit=100000
    )

    packet_records = []
    for packet in packets:
        packet_features = extract_packet_features(packet)
        if packet_features is not None:
            packet_records.append(packet_features)

    packet_df = pd.DataFrame(packet_records)
    packet_df.to_csv(
        os.path.join(OUTPUT_DIR, "packet_features.csv"),
        index=False
    )

    flows = build_flows(packets)
    print(f"[INFO] Total flows: {len(flows)}")

    flow_records = []
    for flow_key, flow_packets in flows.items():
        flow_features = extract_flow_features(flow_packets)
        if flow_features is None:
            continue

        flow_features.update({
            "Src IP": flow_key[0],
            "Dst IP": flow_key[1],
            "Src Port": flow_key[2],
            "Dst Port": flow_key[3],
            "Protocol": flow_key[4],
        })
        flow_features["Flow Key"] = str(flow_key)
        flow_records.append(flow_features)

    flow_df = pd.DataFrame(flow_records)
    flow_df.to_csv(
        os.path.join(OUTPUT_DIR, "flow_features.csv"),
        index=False
    )

    print(f"[INFO] Saved packet features: {len(packet_df)} rows")
    print(f"[INFO] Saved flow features: {len(flow_df)} rows")


if __name__ == "__main__":
    main()
