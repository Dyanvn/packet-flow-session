from src.ingestion.packet_reader import load_packets

packets = load_packets(
    "data/sample/Friday-WorkingHours.pcap",
    limit=100
)

print(len(packets))