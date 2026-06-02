import os
import sys
import argparse

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.ingestion.packet_reader import load_packets
from src.features.packet.packet_features import extract_packet_features

OUTPUT_DIR = "data/processed"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "packet_model_dataset.csv")


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def infer_packet_label(features):
    if features.get("Protocol") == "TCP" and features.get("SYN Flag") == 1 and features.get("ACK Flag") == 0:
        return 1
    return 0


def build_packet_dataset(pcap_path, label=None, heuristic=False, limit=None):
    packets = load_packets(pcap_path, limit=limit or 100000)
    records = []
    for packet in packets:
        features = extract_packet_features(packet)
        if features is None:
            continue
        if heuristic:
            features["Label"] = infer_packet_label(features)
        else:
            features["Label"] = label
        records.append(features)

    if not records:
        raise ValueError(f"No packet features extracted from {pcap_path}")

    import pandas as pd
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[INFO] Saved packet model dataset: {OUTPUT_PATH}")
    print(f"[INFO] Rows: {len(df)}, Columns: {len(df.columns)}")
    print(f"[INFO] Label distribution:\n{df['Label'].value_counts().to_string()}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Build packet-level model dataset from a PCAP file.")
    parser.add_argument("--pcap", default="data/sample/Friday-WorkingHours.pcap", help="PCAP file to extract packet features from")
    parser.add_argument("--label", type=int, choices=[0, 1], help="Fixed label to assign to all extracted packets")
    parser.add_argument("--heuristic", action="store_true", help="Infer packet labels heuristically from flags")
    parser.add_argument("--limit", type=int, default=100000, help="Maximum number of packets to read")
    args = parser.parse_args()

    if args.label is None and not args.heuristic:
        args.heuristic = True

    ensure_output_dir()
    build_packet_dataset(args.pcap, label=args.label, heuristic=args.heuristic, limit=args.limit)


if __name__ == '__main__':
    main()
