import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.pipeline.build_packet_dataset import build_packet_dataset
from src.models.packet_model.train_xgboost import main as train_packet_model
from src.models.packet_model.evaluate_model import main as evaluate_packet_model


def main():
    print('[INFO] Starting full packet pipeline...')
    build_packet_dataset('data/sample/Friday-WorkingHours.pcap', heuristic=True)
    try:
        train_packet_model()
    except ValueError as exc:
        print(f'[WARN] Packet training skipped: {exc}')
    try:
        evaluate_packet_model()
    except ValueError as exc:
        print(f'[WARN] Packet evaluation skipped: {exc}')
    print('[INFO] Packet pipeline complete.')


if __name__ == '__main__':
    main()
