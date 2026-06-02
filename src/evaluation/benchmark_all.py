import os
import pandas as pd

# This script runs the existing per-model benchmark scripts (flow and packet)
# and aggregates their CSV outputs into a single baseline report.

from src.evaluation.benchmark_model import main as benchmark_flow
from src.evaluation.benchmark_packet import main as benchmark_packet

FLOW_CSV = "outputs/metrics/flow_performance_report.csv"
PACKET_CSV = "outputs/metrics/packet_performance_report.csv"
OUT_BASELINE = "outputs/metrics/benchmark_all_baseline.csv"
OUT_COMPARISON = "outputs/metrics/benchmark_comparison.csv"


def ensure_dirs():
    os.makedirs(os.path.dirname(OUT_BASELINE), exist_ok=True)


def load_if_exists(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def main():
    ensure_dirs()

    print('[INFO] Running flow benchmark...')
    try:
        benchmark_flow()
    except Exception as e:
        print('[WARN] Flow benchmark failed:', e)

    print('[INFO] Running packet benchmark...')
    try:
        benchmark_packet()
    except Exception as e:
        print('[WARN] Packet benchmark failed:', e)

    flow_df = load_if_exists(FLOW_CSV)
    packet_df = load_if_exists(PACKET_CSV)

    frames = []
    if flow_df is not None:
        flow_df = flow_df.copy()
        flow_df['model'] = 'flow'
        frames.append(flow_df)
    if packet_df is not None:
        packet_df = packet_df.copy()
        packet_df['model'] = 'packet'
        frames.append(packet_df)

    if not frames:
        raise RuntimeError('No benchmark outputs found for either flow or packet')

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(OUT_BASELINE, index=False)
    print(f'[INFO] Baseline combined benchmark saved to {OUT_BASELINE}')

    # Simple comparison: pivot key metrics by model
    metrics = ['inference_time_seconds', 'throughput_samples_per_second', 'avg_latency_milliseconds']
    compare = combined[['model'] + [m for m in metrics if m in combined.columns]].groupby('model').mean()
    compare.to_csv(OUT_COMPARISON)
    print(f'[INFO] Comparison summary saved to {OUT_COMPARISON}')
    print(compare)


if __name__ == '__main__':
    main()
