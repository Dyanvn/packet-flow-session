from collections import defaultdict

from scapy.layers.inet import IP
from scapy.layers.inet import TCP
from scapy.layers.inet import UDP


def build_flows(packets):

    flows = defaultdict(list)

    for packet in packets:

        if IP not in packet:
            continue

        protocol = None
        sport = 0
        dport = 0

        if TCP in packet:

            protocol = "TCP"

            sport = packet[TCP].sport

            dport = packet[TCP].dport

        elif UDP in packet:

            protocol = "UDP"

            sport = packet[UDP].sport

            dport = packet[UDP].dport

        else:
            continue

        flow_key = (

            packet[IP].src,
            packet[IP].dst,

            sport,
            dport,

            protocol
        )

        flows[flow_key].append(packet)

    return flows