from scapy.layers.inet import IP
from scapy.layers.inet import TCP
from scapy.layers.inet import UDP


def extract_packet_features(packet):
    if IP not in packet:
        return None

    features = {
        "Timestamp": float(packet.time),
        "Src IP": packet[IP].src,
        "Dst IP": packet[IP].dst,
        "Protocol": None,
        "Src Port": 0,
        "Dst Port": 0,
        "Packet Length": len(packet),
        "SYN Flag": 0,
        "ACK Flag": 0,
        "FIN Flag": 0,
        "RST Flag": 0,
        "PSH Flag": 0,
        "URG Flag": 0,
    }

    if TCP in packet:
        features["Protocol"] = "TCP"
        features["Src Port"] = packet[TCP].sport
        features["Dst Port"] = packet[TCP].dport
        flags = str(packet[TCP].flags)
        features["SYN Flag"] = 1 if "S" in flags else 0
        features["ACK Flag"] = 1 if "A" in flags else 0
        features["FIN Flag"] = 1 if "F" in flags else 0
        features["RST Flag"] = 1 if "R" in flags else 0
        features["PSH Flag"] = 1 if "P" in flags else 0
        features["URG Flag"] = 1 if "U" in flags else 0
    elif UDP in packet:
        features["Protocol"] = "UDP"
        features["Src Port"] = packet[UDP].sport
        features["Dst Port"] = packet[UDP].dport

    return features
