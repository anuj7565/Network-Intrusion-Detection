import pandas as pd
import argparse
import time
import datetime
import preprocessing
import model

def run_realtime_simulation():
    parser = argparse.ArgumentParser(description="Real-time Intrusion Detection Simulation")
    parser.add_argument("--data", required=True, help="Path to the dataset file")
    parser.add_argument("--model", choices=['dt', 'rf'], required=True, help="Model to use for simulation")
    args = parser.parse_args()

    columns = ['duration','protocol_type','service','flag','src_bytes',
    'dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins',
    'logged_in','num_compromised','root_shell','su_attempted','num_root',
    'num_file_creations','num_shells','num_access_files','num_outbound_cmds',
    'is_host_login','is_guest_login','count','srv_count','serror_rate',
    'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
    'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
    'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate',
    'dst_host_srv_rerror_rate','label']

    print("--- Loading and Preprocessing Data ---")
    df = pd.read_csv(args.data, names=columns, header=None)
    df = preprocessing.simplify_labels(df)
    df = preprocessing.encode_features(df)
    df, _ = preprocessing.scale_features(df)
    
    X_train, X_test, y_train, y_test = preprocessing.split_data(df)

    print(f"--- Training {args.model} model ---")
    if args.model == 'dt':
        trained_model = model.train_decision_tree(X_train, y_train)
    else:
        trained_model = model.train_random_forest(X_train, y_train)

    print("\n--- Starting Real-time Simulation ---")
    print("Row | Timestamp | Prediction")
    print("-" * 40)

    for i in range(len(X_test)):
        row = X_test.iloc[[i]]
        prediction = trained_model.predict(row)[0]
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        label_str = "ATTACK" if prediction == 1 else "NORMAL"
        print(f"{i+1:3} | {timestamp} | {label_str}")
        
        time.sleep(1)

if __name__ == "__main__":
    run_realtime_simulation()
