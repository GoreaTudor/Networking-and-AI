import pandas as pd
from scapy.all import rdpcap

from data_analysis.constants import get_protocol

FILE_NAME = "..\\data\\test_dos2.pcap"

if __name__ == '__main__':
    packets = rdpcap(FILE_NAME)

    data = []
    start_time = packets[0].time if len(packets) > 0 else 0

    for pkt in packets:
        entry = {
            "time": pkt.time - start_time,
            "size": len(pkt),
        }

        if pkt.haslayer("IP"):
            entry["protocol"] = get_protocol(pkt["IP"].proto)
            entry["src_IP"] = pkt["IP"].src
            entry["dst_IP"] = pkt["IP"].dst

        elif pkt.haslayer("TCP"):
            entry["protocol"] = get_protocol(pkt["TCP"].proto)
            entry["src_port"] = pkt["TCP"].sport
            entry["dst_port"] = pkt["TCP"].dport
            entry["flags"] = pkt["TCP"].flags

        elif pkt.haslayer("UDP"):
            entry["protocol"] = get_protocol(pkt["UDP"].proto)
            entry["src_port"] = pkt["UDP"].sport
            entry["dst_port"] = pkt["UDP"].dport

        data.append(entry)

    df = pd.DataFrame(data)
    print(df)
