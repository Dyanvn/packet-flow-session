from scapy.layers.inet import TCP


def extract_flow_features(flow_packets):

    total_packets = len(flow_packets)

    if total_packets == 0:

        return None

    timestamps = []

    packet_sizes = []

    syn_count = 0

    ack_count = 0

    for packet in flow_packets:

        timestamps.append(float(packet.time))

        packet_sizes.append(len(packet))

        if TCP in packet:

            flags = str(packet[TCP].flags)

            if "S" in flags:
                syn_count += 1

            if "A" in flags:
                ack_count += 1

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

        "Total Backward Packets": 0,

        "SYN Flag Count": syn_count,

        "ACK Flag Count": ack_count,

        "Average Packet Size": avg_packet_size,

        "Packet Length Mean": packet_length_mean
    }

    return features