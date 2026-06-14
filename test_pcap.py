from scapy.utils import PcapReader

count = 0

try:
    for pkt in PcapReader("data/sample/attack_demo.pcap"):
        count += 1

        if count % 1000 == 0:
            print("Đã đọc:", count)

    print("Tổng packet:", count)

except Exception as e:
    print("Lỗi:", e)