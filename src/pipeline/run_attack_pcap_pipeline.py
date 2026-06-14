# src/pipeline/run_attack_pcap_pipeline.py

from pathlib import Path
import pandas as pd
from scapy.utils import rdpcap  # Sử dụng bộ đọc mặc định của Scapy nếu chưa viết file riêng

# Import chuẩn xác theo cấu trúc file của bạn
from src.features.packet.packet_features import extract_packet_features
from src.features.flow.flow_builder import build_flows
from src.features.flow.flow_features import extract_flow_features
from src.features.session.session_features import build_session_dataset


def run_attack_pcap_pipeline(
    pcap_path: str | Path = "data/sample/attack_3h45.pcap",
    packet_csv_out: str | Path = "data/processed/packet_attack_dataset.csv",
    flow_csv_out: str | Path = "data/processed/flow_attack_dataset.csv",
    session_csv_out: str | Path = "data/processed/session_attack_dataset.csv",
):
    pcap_path = Path(pcap_path)
    packet_csv_out = Path(packet_csv_out)
    flow_csv_out = Path(flow_csv_out)
    session_csv_out = Path(session_csv_out)

    if not pcap_path.exists():
        raise FileNotFoundError(f"Không tìm thấy PCAP tấn công: {pcap_path}")

    print(f"[AttackPipeline] Đọc PCAP tấn công từ: {pcap_path}")

    # 1) Đọc file PCAP gốc thành danh sách các Scapy packets
    try:
        # Nếu bạn có hàm custom trong packet_reader, thay rdpcap bằng hàm đó
        packets = rdpcap(str(pcap_path))
    except Exception as e:
        print(f"Lỗi khi đọc file PCAP qua Scapy: {e}")
        return
        
    print(f"[AttackPipeline] Số packet đọc được: {len(packets)}")

    # 2) Trích xuất Packet-level features
    # Vì extract_packet_features nhận vào 1 packet đơn lẻ, ta duyệt qua toàn bộ list packet
    packet_features_list = []
    for pkt in packets:
        feat = extract_packet_features(pkt)
        if feat is not None:
            packet_features_list.append(feat)
            
    df_packet = pd.DataFrame(packet_features_list)
    packet_csv_out.parent.mkdir(parents=True, exist_ok=True)
    df_packet.to_csv(packet_csv_out, index=False)
    print(f"[AttackPipeline] Đã tạo packet_attack_dataset: {packet_csv_out} {df_packet.shape}")

    # 3) Xây dựng Flow và trích xuất Flow-level features từ danh sách packets
    # Hàm build_flows nhận vào list raw packets và trả về một dict thông tin luồng
    flows_dict = build_flows(packets)
    
    flow_features_list = []
    for flow_key, flow_pkts in flows_dict.items():
        flow_feat = extract_flow_features(flow_pkts)
        if flow_feat is not None:
            # Thêm thông tin định danh flow_key nếu cần thiết cho việc tracking
            flow_feat["src_ip"] = flow_key[0]
            flow_feat["dst_ip"] = flow_key[1]
            # Mặc định dữ liệu attack mới chưa có nhãn, gán tạm thời = 0 để hàm Session không bị lỗi key
            flow_feat["Label"] = 0 
            flow_features_list.append(flow_feat)
            
    df_flow = pd.DataFrame(flow_features_list)
    flow_csv_out.parent.mkdir(parents=True, exist_ok=True)
    df_flow.to_csv(flow_csv_out, index=False)
    print(f"[AttackPipeline] Đã tạo flow_attack_dataset: {flow_csv_out} {df_flow.shape}")

    # 4) Tạo Session-level dataset từ Flow dataset vừa sinh ra
    # Hàm build_session_dataset sẽ đọc lại file flow_csv_out, chia window và lưu kết quả xuống session_csv_out
    build_session_dataset(
        flow_csv_path=flow_csv_out,
        output_csv_path=session_csv_out,
        window_size_sec=100,
    )
    print(f"[AttackPipeline] Đã tạo session_attack_dataset: {session_csv_out}")

    print("[AttackPipeline] PIPELINE THÀNH CÔNG.")


if __name__ == "__main__":
    run_attack_pcap_pipeline()