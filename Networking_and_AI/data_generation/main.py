from generator import generate_traffic

if __name__ == '__main__':
    default_time = 5
    normal = ("normal", 0.01)

    scenarios = {
        "": [],
        "normal": [normal],

        # Simple DOS
        "ping_flood_1": [("ping_flood", default_time)],
        "ping_flood_2": [normal, ("ping_flood", default_time), normal],

        "syn_flood_1": [("syn_flood", default_time)],
        "syn_flood_2": [normal, ("syn_flood", default_time), normal],

        "udp_flood_1": [("udp_flood", default_time)],
        "udp_flood_2": [normal, ("udp_flood", default_time), normal],

        # D-DOS
        "ping_ddos_1": [("ping_ddos", default_time)],
        "ping_ddos_2": [normal, ("ping_ddos", default_time), normal],

        "syn_ddos_1": [("syn_ddos", default_time)],
        "syn_ddos_2": [normal, ("syn_ddos", default_time), normal],

        "udp_ddos_1": [("udp_ddos", default_time)],
        "udp_ddos_2": [normal, ("udp_ddos", default_time), normal],

        # Other
        "rdos_1": [("reflected_dos", default_time)],
        "rdos_2": [normal, ("reflected_dos", default_time), normal],

        "pscan_1": [("port_scan", default_time)],
        "pscan_2": [normal, ("port_scan", default_time), normal],
    }

    generate_traffic(scenarios["pscan_1"])
