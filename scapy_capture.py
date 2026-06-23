from scapy.all import sniff, IP, TCP, UDP

# Configuration
INTERFACE = "eth0"
PROTOCOL_MAP = {6: "TCP", 17: "UDP", 1: "ICMP"}
def process_packet(packet):
    if not packet.haslayer(IP):
        return

    ip_layer = packet[IP]
    
    
    # Extract transport layer info
    src_port = None
    dst_port = None
    tcp_flags = None
    protocol = PROTOCOL_MAP.get(ip_layer.proto, ip_layer.proto)

    if packet.haslayer(TCP):
        src_port = packet[TCP].sport
        dst_port = packet[TCP].dport
        tcp_flags = packet[TCP].flags
    elif packet.haslayer(UDP):
        src_port = packet[UDP].sport
        dst_port = packet[UDP].dport

    print(f"Protocol: {protocol}")
    print(f"Source IP: {ip_layer.src}")
    print(f"Destination IP: {ip_layer.dst}")
    print(f"Source Port: {src_port}")
    print(f"Destination Port: {dst_port}")
    print(f"Packet Size: {len(packet)} bytes")
    print(f"TCP Flags: {tcp_flags}")
    print("-" * 20)

if __name__ == "__main__":
    sniff(iface=INTERFACE, prn=process_packet, count=20)
