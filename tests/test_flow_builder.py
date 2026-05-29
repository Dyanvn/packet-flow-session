from src.ingestion.packet_reader import load_packets

from src.features.flow.flow_builder import (

    build_flows
)

packets = load_packets(

    "data/sample/Friday-WorkingHours.pcap",

    limit=1000
)

flows = build_flows(packets)

print("Number of flows:", len(flows))