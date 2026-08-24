import sqlite3
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
from scapy_capture import predict_flow

def init_db():
    conn = sqlite3.connect("nids.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            src_ip TEXT NOT NULL,
            dst_ip TEXT NOT NULL,
            duration REAL NOT NULL,
            protocol_type TEXT NOT NULL,
            service TEXT NOT NULL,
            flag TEXT NOT NULL,
            predicted_label TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

app = FastAPI()

class FlowFeatures(BaseModel):
    duration: float
    protocol_type: Literal["tcp", "udp", "icmp"]
    service: Literal["http", "http_443", "ssh", "ftp", "telnet", "smtp",
                      "domain_u", "pop_3", "imap4", "ntp_u", "other"]
    src_bytes: int
    dst_bytes: int
    flag: Literal["REJ", "SF", "S0", "OTH"]
    count: int
    same_srv_rate: float
    serror_rate: float
    rerror_rate: float
    diff_srv_rate: float
    land: int
    wrong_fragment: int
    urgent: int

    hot: int = 0
    num_failed_logins: int = 0
    logged_in: int = 0
    num_compromised: int = 0
    root_shell: int = 0
    su_attempted: int = 0
    num_root: int = 0
    num_file_creations: int = 0
    num_shells: int = 0
    num_access_files: int = 0
    num_outbound_cmds: int = 0
    is_host_login: int = 0
    is_guest_login: int = 0
    dst_host_count: int = 0
    dst_host_srv_count: int = 0
    dst_host_same_srv_rate: float = 0
    dst_host_diff_srv_rate: float = 0
    dst_host_same_src_port_rate: float = 0
    dst_host_srv_diff_host_rate: float = 0
    dst_host_serror_rate: float = 0
    dst_host_srv_serror_rate: float = 0
    dst_host_rerror_rate: float = 0
    dst_host_srv_rerror_rate: float = 0
    srv_count: int = 0
    srv_serror_rate: float = 0
    srv_rerror_rate: float = 0
    srv_diff_host_rate: float = 0

@app.post("/predict")
def predict(features: FlowFeatures):
    result = predict_flow(features.dict())
    label = "ATTACK" if int(result) == 1 else "NORMAL"
    return {"prediction": label}
