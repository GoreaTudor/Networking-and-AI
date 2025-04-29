from generator import generate_traffic

if __name__ == '__main__':
    default_time = 5
    normal = ("normal", 0.01)

    scenarios = [
        [],         # 0
        [normal],   # 1

        [("simple_dos", default_time)],                     # 2
        [normal, ("simple_dos", default_time), normal],     # 3

        [("ddos", default_time)],                           # 4
        [normal, ("ddos", default_time), normal],           # 5

        [("reflected_dos", 20)],                            # 6
        [normal, ("reflected_dos", default_time), normal],  # 7

        [("port_scan", default_time)],                      # 8
        [normal, ("port_scan", default_time), normal],      # 9
    ]

    generate_traffic(scenarios[8])
