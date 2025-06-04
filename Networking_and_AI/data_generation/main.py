from generator import generate_traffic

normal_time = 0.01
default_time = 5

normal = ("normal", normal_time)
ping_flood = ("ping_flood", default_time)
syn_flood = ("syn_flood", 0.1)
fin_flood = ("fin_flood", 0.1)
rst_flood = ("rst_flood", 0.1)
udp_flood = ("udp_flood", 0.1)
ping_ddos = ("ping_ddos", default_time)
syn_ddos = ("syn_ddos", 0.1)
fin_ddos = ("fin_ddos", 0.1)
rst_ddos = ("rst_ddos", 0.1)
udp_ddos = ("udp_ddos", 0.1)
reflected_dos = ("reflected_dos", 20)
port_scan = ("port_scan", default_time)

scenarios = {
    "": [],
    "normal": [normal],

    # Simple DOS
    "ping_flood": [normal, ping_flood, normal],
    "syn_flood": [normal, syn_flood, normal],
    "fin_flood": [normal, fin_flood, normal],
    "rst_flood": [normal, rst_flood, normal],
    "udp_flood": [normal, udp_flood, normal],

    # D-DOS
    "ping_ddos": [normal, ping_ddos, normal],
    "syn_ddos": [normal, syn_ddos, normal],
    "fin_ddos": [normal, fin_ddos, normal],
    "rst_ddos": [normal, rst_ddos, normal],
    "udp_ddos": [normal, udp_ddos, normal],

    # Other
    "rdos": [normal, reflected_dos, normal],
    "pscan": [normal, port_scan, normal],

    # all types
    "all_flood": [normal, ping_flood, normal, syn_flood, normal, udp_flood, normal],
    "all_ddos": [normal, ping_ddos, normal, syn_ddos, normal, udp_ddos, normal],
}

scenarios_0 = {
    "": [],
    "normal": [("normal", 1)],

    # Simple DOS
    "ping_flood": [("ping_flood", 60)],
    "syn_flood": [("syn_flood", 3)],
    "fin_flood": [("fin_flood", 3)],
    "rst_flood": [("rst_flood", 3)],
    "udp_flood": [("udp_flood", 3)],

    # D-DOS
    "ping_ddos": [("ping_ddos", 20)],
    "syn_ddos": [("syn_ddos", 1)],
    "fin_ddos": [("fin_ddos", 1)],
    "rst_ddos": [("rst_ddos", 1)],
    "udp_ddos": [("udp_ddos", 1)],

    # Other
    "rdos": [("reflected_dos", 60)],
    "pscan": [("port_scan", 20)],
}

if __name__ == '__main__':
    generate_traffic(scenarios_0["pscan"])
