from scapy.all import PcapReader


def load_packets(pcap_file, limit=10000):

    packets = []

    print(f"[INFO] Reading {pcap_file}")

    with PcapReader(pcap_file) as pcap:

        for i, packet in enumerate(pcap):

            packets.append(packet)

            if i + 1 >= limit:
                break

    print(f"[INFO] Loaded {len(packets)} packets")

    return packets