from generator import generate_traffic

normal_time = 0.01
default_time = 5

normal = ("normal", normal_time)
ping_flood = ("ping_flood", default_time)
syn_flood = ("syn_flood", 0.1)
udp_flood = ("udp_flood", 0.1)
ping_ddos = ("ping_ddos", default_time)
syn_ddos = ("syn_ddos", 0.1)
udp_ddos = ("udp_ddos", 0.1)
reflected_dos = ("reflected_dos", 20)
port_scan = ("port_scan", default_time)

scenarios = {
    "": [],
    "normal": [normal],

    # Simple DOS
    "ping_flood": [normal, ping_flood, normal],
    "syn_flood": [normal, syn_flood, normal],
    "udp_flood": [normal, udp_flood, normal],

    # D-DOS
    "ping_ddos": [normal, ping_ddos, normal],
    "syn_ddos": [normal, syn_ddos, normal],
    "udp_ddos": [normal, udp_ddos, normal],

    # Other
    "rdos": [normal, reflected_dos, normal],
    "pscan": [normal, port_scan, normal],

    # all types
    "all_flood": [normal, ping_flood, normal, syn_flood, normal, udp_flood, normal],
    "all_ddos": [normal, ping_ddos, normal, syn_ddos, normal, udp_ddos, normal],
}

if __name__ == '__main__':
    generate_traffic(scenarios["rdos"])
