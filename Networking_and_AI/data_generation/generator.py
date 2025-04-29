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
        writer.writerow(["attack_type", "start_time", "end_time", "attacker", "victim"])
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


def simple_dos(victim, attacker, duration):
    print("\n[*] Starting simple DoS (ping flood)...")
    cmd = "ping -f -c " + str(duration * 10) + " " + victim.IP() + " &"
    print("-- Attacker: " + cmd)
    attacker.cmd(cmd)
    time.sleep(duration)


def ddos(victim, attackers, duration):
    print("\n[*] Starting DDoS (multi-host ping flood)...")
    for attacker in attackers:
        cmd = "ping -f -c " + str(duration * 10) + " " + victim.IP() + " &"
        print("-- Attacker " + attacker.name + ": " + cmd)
        attacker.cmd(cmd)
    time.sleep(duration)


# DOES NOT WORK
def reflected_dos(victim, attacker, dns_server, duration):
    print("\n[*] Starting reflected DoS using spoofed DNS requests...")
    # dns_cmd = "dnsmasq --no-daemon --log-queries --log-facility=- &"
    # print("-- DNS Server: " + dns_cmd)

    # Start fake DNS server on dns host (responds to dig queries)
    dns_server.cmd("nohup python3 -m http.server 53 >/dev/null 2>&1 &")  # dummy server using port 53

    # Use scapy or hping3 to send spoofed UDP packets to port 53
    cmd = "hping3 --udp -a " + victim.IP() + " -p 53 -i u10000 " + dns_server.IP() + " &"
    print("-- Attacker: " + cmd)
    attacker.cmd(cmd)
    time.sleep(duration)


# DOES NOT WORK
def port_scan(victim, attacker, duration):
    print("\n[*] Starting port scan...")
    victim.cmd("nohup nc -lkp 80 >/dev/null 2>&1 &")
    victim.cmd("nohup nc -lkp 443 >/dev/null 2>&1 &")

    cmd = "nmap -sS -p 80,443 " + victim.IP() + " &"
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
            attackers = "h2"
        elif attack_type == "simple_dos":
            simple_dos(victim=h1, attacker=a1, duration=duration)
            attackers = "a1"
        elif attack_type == "ddos":
            ddos(victim=h1, attackers=[a1, a2, a3], duration=duration)
            attackers = "a1,a2,a3"
        elif attack_type == "port_scan":
            port_scan(victim=h1, attacker=a1, duration=duration)
            attackers = "a1"
        elif attack_type == "reflected_dos":
            reflected_dos(victim=h1, attacker=a1, dns_server=dns, duration=duration)
            attackers = "a1"
        else:
            print("[!] Unknown attack type: " + attack_type)
            continue

        end = time.time() - attack_start
        metadata.append([attack_type, round(start, 2), round(end, 2), attackers, "h1"])

    print("\n[*] Stopping tcpdump...")
    h1.cmd("pkill tcpdump")
    time.sleep(1)

    write_metadata()

    net.stop()
