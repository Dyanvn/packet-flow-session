# src/pipeline/run_attack_monitor.py

import time

from src.pipeline.run_attack_pcap_pipeline import run_attack_pcap_pipeline
from src.evaluation.demo_realtime_console import demo_realtime_console


def loop_monitor(interval_sec: float = 5.0):
    """
    Vòng lặp:
    - Cứ mỗi interval_sec giây:
      + Đọc attack_demo.pcap -> cập nhật packet/flow/session_attack_dataset
      + Chạy realtime console -> thêm alert vào realtime_alerts_log.csv
    UI sẽ đọc file log này và cập nhật.
    """
    while True:
        print("\n[Monitor] === Chu kỳ mới ===")
        print("[Monitor] Chạy pipeline PCAP tấn công -> dataset attack...")
        run_attack_pcap_pipeline()

        print("[Monitor] Chạy realtime console (infer + log alert)...")
        demo_realtime_console()

        print(f"[Monitor] Nghỉ {interval_sec} giây rồi chạy lại...")
        time.sleep(interval_sec)


if __name__ == "__main__":
    loop_monitor(interval_sec=5.0)