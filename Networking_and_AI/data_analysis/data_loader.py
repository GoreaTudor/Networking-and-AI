import pandas as pd
from pandas import DataFrame
from scapy.all import rdpcap

from data_analysis.constants import get_protocol

__PCAP_FILE_PATH = "..\\data\\test_dos2.pcap"
__META_FILE_PATH = "..\\data\\test_dos2.csv"


def load_packets_supervised_data(pcap_file_path=__PCAP_FILE_PATH,
                                 meta_file_path=__META_FILE_PATH,
                                 debug_mode=False) -> DataFrame:
    packets = rdpcap(pcap_file_path)
    metadata_df = pd.read_csv(meta_file_path)


    def get_attack_from_meta(p_time: float) -> str:
        for _, row in metadata_df.iterrows():
            if row["start_time"] <= p_time < row["end_time"]:
                return row["attack_type"]

        return "unknown"


    data = []
    start_time = packets[0].time if len(packets) > 0 else 0

    for pkt in packets:
        packet_time = pkt.time - start_time

        entry = {
            "time": packet_time,
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

        entry["attack_type"] = get_attack_from_meta(packet_time)
        data.append(entry)

    df = pd.DataFrame(data)

    if debug_mode:
        print("\nMetadata values:")
        print(metadata_df)

        print("\nDF:")
        print(df)

    return df


if __name__ == '__main__':
    df = load_packets_supervised_data(debug_mode=True)

    # checks if the labels are added correctly
    # there also seems to be a delay issue with the timing generated in csv
    for i in range(0, 1100, 100):
        print(f"\n{i}: {df.iloc[i].attack_type}")
