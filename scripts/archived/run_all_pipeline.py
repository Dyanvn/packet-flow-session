import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.pipeline.run_flow_pipeline import main as run_flow_pipeline
from src.pipeline.run_packet_pipeline import main as run_packet_pipeline


def main():
    print('[INFO] Starting full packet+flow pipeline...')
    run_flow_pipeline()
    run_packet_pipeline()
    print('[INFO] Full packet+flow pipeline completed.')


if __name__ == '__main__':
    main()
