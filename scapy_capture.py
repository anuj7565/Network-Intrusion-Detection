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

def extract_features(flow_key, packets, completed_flows):
    src_ip_key, src_port_key, dst_ip_key, dst_port_key, proto = flow_key
    
    # 1. Duration
    duration = packets[-1].time - packets[0].time
    
    # 2. Protocol Type
    proto_name = PROTOCOL_MAP.get(proto, str(proto))
    
    # 3. Bytes
    src_bytes = 0
    dst_bytes = 0
    for p in packets:
        if p[IP].src == src_ip_key:
            src_bytes += len(p)
        else:
            dst_bytes += len(p)
            
    # 4. Flags
    has_syn = False
    has_fin = False
    has_rst = False
    
    for p in packets:
        if p.haslayer(TCP):
            f = p[TCP].flags
            if 'S' in f: has_syn = True
            if 'F' in f: has_fin = True
            if 'R' in f: has_rst = True
            
    if has_rst: flag = "REJ"
    elif has_syn and has_fin: flag = "SF"
    elif has_syn and not has_fin: flag = "S0"
    else: flag = "OTH"

    # 5. Count-based features
    last_time = packets[-1].time
    relevant_flows = [f for f in completed_flows if abs(f['last_time'] - last_time) <= 2.0]
    
    same_ip_flows = [f for f in relevant_flows if f['dst_ip'] == dst_ip_key]
    count = len(same_ip_flows)
    
    same_srv_count = sum(1 for f in same_ip_flows if f['dst_port'] == dst_port_key)
    same_srv_rate = same_srv_count / count if count > 0 else 0.0
    
    return {
        "duration": float(duration),
        "protocol": proto_name,
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "flag": flag,
        "count": count,
        "same_srv_rate": same_srv_rate
    }

if __name__ == "__main__":
    sniff(iface=INTERFACE, prn=process_packet, count=100)
    
    # Build completed_flows list
    completed_flows = []
    for flow_key, packets in flows.items():
        _, _, dst_ip, dst_port, _ = flow_key
        completed_flows.append({
            "dst_ip": dst_ip,
            "dst_port": dst_port,
            "last_time": packets[-1].time
        })
    
    print("\nFlow Summary and Features:")
    print("-" * 60)
    for flow_key, packets in flows.items():
        features = extract_features(flow_key, packets, completed_flows)
        src_ip, src_port, dst_ip, dst_port, _ = flow_key
        
        print(f"Flow: {src_ip}:{src_port} <-> {dst_ip}:{dst_port}")
        print(f"  Duration: {features['duration']:.4f}s")
        print(f"  Protocol: {features['protocol']}")
        print(f"  Bytes: Src={features['src_bytes']}, Dst={features['dst_bytes']}")
        print(f"  Flag: {features['flag']}")
        print(f"  Count: {features['count']}, Same Srv Rate: {features['same_srv_rate']:.2f}")
        print("-" * 60)
