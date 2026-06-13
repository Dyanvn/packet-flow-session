import os
import pandas as pd
from src.evaluation.parallel_inference import run_parallel_inference

# For flow dataset/models
FLOW_DATA = "data/processed/flow_model_dataset.csv"
FLOW_MODEL = "outputs/models/xgboost_flow_model.pkl"
FLOW_OUT = "outputs/metrics/flow_parallel_report.csv"

# For packet dataset/models
PACKET_DATA = "data/processed/packet_model_dataset.csv"
PACKET_MODEL = "outputs/models/xgboost_packet_model.pkl"
PACKET_OUT = "outputs/metrics/packet_parallel_report.csv"

WORKERS = [2, 4, 8]


def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def load_flow():
    df = pd.read_csv(FLOW_DATA)
    from src.preprocessing.feature_selection import selected_features
    X = df[selected_features]
    y = df['Label']
    return X, y


def load_packet():
    df = pd.read_csv(PACKET_DATA)
    # map protocol and select features (same as benchmark_packet)
    PROTOCOL_MAP = {"TCP": 1, "UDP": 0}
    if "Protocol" in df.columns:
        df["Protocol Code"] = df["Protocol"].map(PROTOCOL_MAP).fillna(2).astype(int)
    else:
        df["Protocol Code"] = 2

    selected = [
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
    X = df[selected]
    y = df['Label']
    return X, y


def run_for_model(name, X, model_path, out_path):
    ensure_dir(out_path)
    records = []
    for w in WORKERS:
        metrics = run_parallel_inference(model_path, X, workers=w)
        metrics['model'] = name
        metrics['workers'] = w
        records.append(metrics)
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"[INFO] Parallel benchmark for {name} saved to {out_path}")


def main():
    # Flow
    Xf, yf = load_flow()
    run_for_model('flow', Xf, FLOW_MODEL, FLOW_OUT)
    # Packet
    Xp, yp = load_packet()
    run_for_model('packet', Xp, PACKET_MODEL, PACKET_OUT)

    # Combine baseline vs parallel if baseline exists
    baseline = []
    bflow = 'outputs/metrics/flow_performance_report.csv'
    bpacket = 'outputs/metrics/packet_performance_report.csv'
    if os.path.exists(bflow):
        df = pd.read_csv(bflow)
        df['model'] = 'flow'
        df['workers'] = 1
        baseline.append(df)
    if os.path.exists(bpacket):
        df = pd.read_csv(bpacket)
        df['model'] = 'packet'
        df['workers'] = 1
        baseline.append(df)

    parallel_frames = []
    if os.path.exists(FLOW_OUT):
        parallel_frames.append(pd.read_csv(FLOW_OUT))
    if os.path.exists(PACKET_OUT):
        parallel_frames.append(pd.read_csv(PACKET_OUT))

    if baseline and parallel_frames:
        base = pd.concat(baseline, ignore_index=True, sort=False)
        par = pd.concat(parallel_frames, ignore_index=True, sort=False)
        combined = pd.concat([base, par], ignore_index=True, sort=False)
        out_all = 'outputs/metrics/benchmark_post_parallel.csv'
        combined.to_csv(out_all, index=False)
        print(f"[INFO] Combined post-parallel benchmark saved to {out_all}")


if __name__ == '__main__':
    main()
