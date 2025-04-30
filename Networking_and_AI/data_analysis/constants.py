__PROTOCOL_MAP = {
    1: 'ICMP',
    6: 'TCP',
    17: 'UDP',
}

def get_protocol(proto: int):
    return __PROTOCOL_MAP.get(proto, f"Unknown-{proto}")