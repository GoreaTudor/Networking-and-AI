import csv
from scapy.all import rdpcap

__RAW_FILE_PREFIX = ".\\raw\\"
__CONVERTED_FILE_PREFIX = ".\\converted\\"
__UNKNOWN_LABEL = "unknown"
__NORMAL_LABEL = "normal"

__ATTACK_IPS = [
    "10.0.0.10",
    "10.0.0.11",
    "10.0.0.12",
    "10.0.0.53",
]

__PROTOCOL_MAP = {
    1: 'ICMP',
    6: 'TCP',
    17: 'UDP',
}


def get_protocol(proto: int):
    return __PROTOCOL_MAP.get(proto, f"Unknown-{proto}")


def get_attack_label(src_IP, dest_IP, attack_label):
    if src_IP == "" or dest_IP == "":
        return __UNKNOWN_LABEL

    if src_IP in __ATTACK_IPS or dest_IP in __ATTACK_IPS:
        return attack_label

    return __NORMAL_LABEL


def transform_single_attacks_to_csv(file_name: str,
                                    attack_label: str):
    raw_file_name = __RAW_FILE_PREFIX + file_name + ".pcap"
    converted_file_name = __CONVERTED_FILE_PREFIX + file_name + ".csv"

    packets = rdpcap(raw_file_name)

    data = []
    start_time = packets[0].time if len(packets) > 0 else 0

    for pkt in packets:
        packet_time = pkt.time - start_time

        entry = {
            "time": packet_time,
            "size": len(pkt),
            "protocol": "",
            "src_IP": "",
            "dst_IP": "",
            "src_port": "",
            "dst_port": "",
            "flags": "",
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

        entry["attack_type"] = get_attack_label(entry["src_IP"], entry["dst_IP"], attack_label)
        data.append(entry)

    # Write data to CSV
    fieldnames = ["time", "size", "protocol", "src_IP", "dst_IP", "src_port", "dst_port", "flags", "attack_type"]
    with open(converted_file_name, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


if __name__ == '__main__':
    transform_single_attacks_to_csv(file_name="udp_ddos_2",
                                    attack_label="udp_ddos")
