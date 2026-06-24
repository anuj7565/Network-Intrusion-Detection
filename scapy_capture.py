from scapy.all import sniff, IP, TCP, UDP

# Configuration
INTERFACE = "eth0"
PROTOCOL_MAP = {6: "TCP", 17: "UDP", 1: "ICMP"}
flows = {}

def process_packet(packet):
    if not packet.haslayer(IP):
        return

    ip_layer = packet[IP]
    
    # Extract transport layer info
    src_port = 0
    dst_port = 0
    protocol = ip_layer.proto

    if packet.haslayer(TCP):
        src_port = packet[TCP].sport
        dst_port = packet[TCP].dport
    elif packet.haslayer(UDP):
        src_port = packet[UDP].sport
        dst_port = packet[UDP].dport

    # Normalize 5-tuple: (ip1, port1, ip2, port2, protocol)
    # Sort (ip, port) pairs to ensure A->B and B->A map to the same key
    endpoint1 = (ip_layer.src, src_port)
    endpoint2 = (ip_layer.dst, dst_port)
    
    if endpoint1 < endpoint2:
        key = (endpoint1[0], endpoint1[1], endpoint2[0], endpoint2[1], protocol)
    else:
        key = (endpoint2[0], endpoint2[1], endpoint1[0], endpoint1[1], protocol)

    if key not in flows:
        flows[key] = []
    flows[key].append(packet)

if __name__ == "__main__":
    sniff(iface=INTERFACE, prn=process_packet, count=20)
    
    print("\nFlow Summary:")
    print("-" * 40)
    for flow_key, packets in flows.items():
        src_ip, src_port, dst_ip, dst_port, proto = flow_key
        proto_name = PROTOCOL_MAP.get(proto, str(proto))
        print(f"Flow: {src_ip}:{src_port} <-> {dst_ip}:{dst_port} ({proto_name})")
        print(f"Packet count: {len(packets)}")
        print("-" * 40)
