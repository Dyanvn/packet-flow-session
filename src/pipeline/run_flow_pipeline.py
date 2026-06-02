import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.pipeline.build_flow_dataset import main as build_flow_dataset
from src.models.flow_model.train_xgboost import main as train_flow_model
from src.models.flow_model.evaluate_model import main as evaluate_flow_model
from src.evaluation.benchmark_model import main as benchmark_flow_model


def main():
    print('[INFO] Starting full flow pipeline...')
    build_flow_dataset()
    train_flow_model()
    evaluate_flow_model()
    benchmark_flow_model()
    print('[INFO] Full flow pipeline completed successfully.')


if __name__ == '__main__':
    main()
