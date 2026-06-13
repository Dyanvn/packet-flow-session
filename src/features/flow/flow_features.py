from scapy.layers.inet import IP
from scapy.layers.inet import TCP
from scapy.layers.inet import UDP


def extract_flow_features(flow_packets):

    total_packets = len(flow_packets)

    if total_packets == 0:

        return None

    timestamps = []

    packet_sizes = []

    syn_count = 0

    ack_count = 0

    if len(flow_packets) == 0:
        return None

    first_packet = flow_packets[0]
    first_sport = 0
    first_dport = 0
    first_protocol = None

    if TCP in first_packet:
        first_protocol = "TCP"
        first_sport = first_packet[TCP].sport
        first_dport = first_packet[TCP].dport
    elif UDP in first_packet:
        first_protocol = "UDP"
        first_sport = first_packet[UDP].sport
        first_dport = first_packet[UDP].dport

    forward_tuple = (
        first_packet[IP].src,
        first_packet[IP].dst,
        first_sport,
        first_dport,
        first_protocol
    )

    backward_packets = 0

    for packet in flow_packets:

        timestamps.append(float(packet.time))

        packet_sizes.append(len(packet))

        if TCP in packet:

            flags = str(packet[TCP].flags)

            if "S" in flags:
                syn_count += 1

            if "A" in flags:
                ack_count += 1

        if IP not in packet:
            continue

        sport = 0
        dport = 0
        protocol = None

        if TCP in packet:
            protocol = "TCP"
            sport = packet[TCP].sport
            dport = packet[TCP].dport
        elif UDP in packet:
            protocol = "UDP"
            sport = packet[UDP].sport
            dport = packet[UDP].dport

        packet_tuple = (
            packet[IP].src,
            packet[IP].dst,
            sport,
            dport,
            protocol
        )

        if packet_tuple != forward_tuple:
            backward_packets += 1

    flow_duration = (

        max(timestamps)
        -
        min(timestamps)
    )

    avg_packet_size = (

        sum(packet_sizes)
        /
        total_packets
    )

    packet_length_mean = avg_packet_size

    features = {

        "Flow Duration": flow_duration,

        "Total Fwd Packets": total_packets,

        "Total Backward Packets": backward_packets,

        "SYN Flag Count": syn_count,

        "ACK Flag Count": ack_count,

        "Average Packet Size": avg_packet_size,

        "Packet Length Mean": packet_length_mean
    }

    return features