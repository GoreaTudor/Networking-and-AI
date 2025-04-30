from generator import generate_traffic

if __name__ == '__main__':
    default_time = 5
    normal = ("normal", 0.01)

    scenarios = {
        "": [],
        "normal": [normal],

        "dos1": [("simple_dos", default_time)],
        "dos2": [normal, ("simple_dos", default_time), normal],

        "ddos1": [("ddos", default_time)],
        "ddos2": [normal, ("ddos", default_time), normal],

        "rdos1": [("reflected_dos", 20)],
        "rdos2": [normal, ("reflected_dos", default_time), normal],

        "pscan1": [("port_scan", default_time)],
        "pscan2": [normal, ("port_scan", default_time), normal],
    }

    generate_traffic(scenarios["dos1"])
