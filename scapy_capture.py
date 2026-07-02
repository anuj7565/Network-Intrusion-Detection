from scapy.all import sniff, IP, TCP, UDP

# Configuration
INTERFACE = "eth0"
PROTOCOL_MAP = {6: "TCP", 17: "UDP", 1: "ICMP"}
SERVICE_MAP = {
    80: "http", 443: "http_443", 22: "ssh", 21: "ftp", 23: "telnet", 
    25: "smtp", 53: "domain_u", 110: "pop_3", 143: "imap4", 
    123: "ntp_u", 67: "domain_u", 68: "domain_u"
}
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

def get_flow_flag(packets):
    has_syn = False
    has_fin = False
    has_rst = False
    
    for p in packets:
        if p.haslayer(TCP):
            f = p[TCP].flags
            if 'S' in f: has_syn = True
            if 'F' in f: has_fin = True
            if 'R' in f: has_rst = True
            
    if has_rst: return "REJ"
    elif has_syn and has_fin: return "SF"
    elif has_syn and not has_fin: return "S0"
    else: return "OTH"

def extract_features(flow_key, packets, completed_flows):
    src_ip_key, src_port_key, dst_ip_key, dst_port_key, proto = flow_key
    
    # 1. Duration
    duration = packets[-1].time - packets[0].time
    
    # 2. Protocol Type
    proto_name = PROTOCOL_MAP.get(proto, str(proto))
    
    # 3. Service
    service = SERVICE_MAP.get(dst_port_key, SERVICE_MAP.get(src_port_key, "other"))
    
    # 4. Bytes
    src_bytes = 0
    dst_bytes = 0
    for p in packets:
        if p[IP].src == src_ip_key:
            src_bytes += len(p)
        else:
            dst_bytes += len(p)
            
    # 5. Flags
    flag = get_flow_flag(packets)

    # 6. Count-based features
    last_time = packets[-1].time
    relevant_flows = [f for f in completed_flows if abs(f['last_time'] - last_time) <= 2.0]
    
    same_ip_flows = [f for f in relevant_flows if f['dst_ip'] == dst_ip_key]
    count = len(same_ip_flows)
    
    same_srv_count = sum(1 for f in same_ip_flows if f['dst_port'] == dst_port_key)
    same_srv_rate = same_srv_count / count if count > 0 else 0.0
    
    # Rate-based features
    serror_rate = sum(1 for f in same_ip_flows if f['flag'] == 'S0') / count if count > 0 else 0.0
    rerror_rate = sum(1 for f in same_ip_flows if f['flag'] == 'REJ') / count if count > 0 else 0.0
    diff_srv_rate = sum(1 for f in same_ip_flows if f['dst_port'] != dst_port_key) / count if count > 0 else 0.0

    # 7. New Features
    land = 1 if (src_ip_key == dst_ip_key and src_port_key == dst_port_key) else 0
    wrong_fragment = sum(1 for p in packets if p[IP].frag > 0)
    urgent = sum(1 for p in packets if p.haslayer(TCP) and 'U' in p[TCP].flags)
    
    return {
        "duration": float(duration),
        "protocol": proto_name,
        "service": service,
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "flag": flag,
        "count": count,
        "same_srv_rate": same_srv_rate,
        "serror_rate": serror_rate,
        "rerror_rate": rerror_rate,
        "diff_srv_rate": diff_srv_rate,
        "land": land,
        "wrong_fragment": wrong_fragment,
        "urgent": urgent,
        "hot": 0,
        "num_failed_logins": 0,
        "logged_in": 0,
        "num_compromised": 0,
        "root_shell": 0,
        "su_attempted": 0,
        "num_root": 0,
        "num_file_creations": 0,
        "num_shells": 0,
        "num_access_files": 0,
        "num_outbound_cmds": 0,
        "is_host_login": 0,
        "is_guest_login": 0,
        "dst_host_count": 0,
        "dst_host_srv_count": 0,
        "dst_host_same_srv_rate": 0,
        "dst_host_diff_srv_rate": 0,
        "dst_host_same_src_port_rate": 0,
        "dst_host_srv_diff_host_rate": 0,
        "dst_host_serror_rate": 0,
        "dst_host_srv_serror_rate": 0,
        "dst_host_rerror_rate": 0,
        "dst_host_srv_rerror_rate": 0,
        "srv_count": 0,
        "srv_serror_rate": 0,
        "srv_rerror_rate": 0,
        "srv_diff_host_rate": 0
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
            "last_time": packets[-1].time,
            "flag": get_flow_flag(packets)
        })
    
    print("\nFlow Summary and Features:")
    print("-" * 60)
    for flow_key, packets in flows.items():
        features = extract_features(flow_key, packets, completed_flows)
        src_ip, src_port, dst_ip, dst_port, _ = flow_key
        
        print(f"Flow: {src_ip}:{src_port} <-> {dst_ip}:{dst_port}")
        print(f"  Duration: {features['duration']:.4f}s")
        print(f"  Protocol: {features['protocol']}, Service: {features['service']}")
        print(f"  Bytes: Src={features['src_bytes']}, Dst={features['dst_bytes']}")
        print(f"  Flag: {features['flag']}")
        print(f"  Count: {features['count']}, Same Srv Rate: {features['same_srv_rate']:.2f}")
        print(f"  Rates: SError={features['serror_rate']:.2f}, RError={features['rerror_rate']:.2f}, DiffSrv={features['diff_srv_rate']:.2f}")
        print(f"  Land: {features['land']}, Wrong Frag: {features['wrong_fragment']}, Urgent: {features['urgent']}")
        print("-" * 60)
