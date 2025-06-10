import csv

from scapy.all import rdpcap
from scapy.layers.inet import IP, TCP, UDP

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

        if IP in pkt:
            entry["src_IP"] = pkt[IP].src
            entry["dst_IP"] = pkt[IP].dst
            entry["protocol"] = get_protocol(pkt[IP].proto)

            if TCP in pkt:
                entry["src_port"] = pkt[TCP].sport
                entry["dst_port"] = pkt[TCP].dport
                entry["flags"] = pkt[TCP].flags

            elif UDP in pkt:
                entry["src_port"] = pkt[UDP].sport
                entry["dst_port"] = pkt[UDP].dport

        entry["attack_type"] = get_attack_label(entry["src_IP"], entry["dst_IP"], attack_label)
        data.append(entry)

    # Write data to CSV
    fieldnames = ["time", "size", "protocol", "src_IP", "dst_IP", "src_port", "dst_port", "flags", "attack_type"]
    with open(converted_file_name, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def convert_all_files():
    all_files = [
        ("normal_0", "unknown"),
        ("normal_1", "unknown"),

        # floods
        ("ping_flood_0", "ping_flood"),  # ping
        ("ping_flood_1", "ping_flood"),
        ("syn_flood_0", "syn_flood"),  # syn
        ("syn_flood_1", "syn_flood"),
        ("fin_flood_0", "fin_flood"),  # fin
        ("fin_flood_1", "fin_flood"),
        ("rst_flood_0", "rst_flood"),  # rst
        ("rst_flood_1", "rst_flood"),
        ("udp_flood_0", "udp_flood"),  # udp
        ("udp_flood_1", "udp_flood"),

        # ddos
        ("ping_ddos_0", "ping_ddos"),  # ping
        ("ping_ddos_1", "ping_ddos"),
        ("syn_ddos_0", "syn_ddos"),  # syn
        ("syn_ddos_1", "syn_ddos"),
        ("fin_ddos_0", "fin_ddos"),  # fin
        ("fin_ddos_1", "fin_ddos"),
        ("rst_ddos_0", "rst_ddos"),  # rst
        ("rst_ddos_1", "rst_ddos"),
        ("udp_ddos_0", "udp_ddos"),  # udp
        ("udp_ddos_1", "udp_ddos"),

        # other
        ("reflected_dos_0", "reflected_dos"),  # reflected dos
        ("reflected_dos_1", "reflected_dos"),
        ("pscan_test", "port_scan"),  # pscan
        ("pscan_0", "port_scan"),
        ("pscan_1", "port_scan"),
    ]

    for file_name, attack_label in all_files:
        print(">> " + file_name)
        transform_single_attacks_to_csv(file_name=file_name,
                                        attack_label=attack_label)


if __name__ == '__main__':
    ### do all files:
    convert_all_files()

    ### specific file:
    # transform_single_attacks_to_csv(file_name="ping_flood_1",
    #                                 attack_label="ping_flood")
