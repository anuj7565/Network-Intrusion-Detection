import pandas as pd
import argparse
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score
import preprocessing
import model
import clustering
import datetime

def generate_report(df, dt_model, rf_model, X_test, y_test):
    """
    Generates a text report summarizing the model performance and dataset statistics.

    Documentation is critical in security tools because it provides an audit trail 
    of how decisions were made, ensures reproducibility of results, and allows 
    security analysts to understand the model's limitations and performance 
    characteristics when investigating potential breaches.
    """
    print("--- Generating report.txt ---")
    
    with open('report.txt', 'w') as f:
        f.write(f"Intrusion Detection System Report\n")
        f.write(f"Generated on: {datetime.datetime.now()}\n\n")
        f.write(f"Dataset Statistics:\n")
        f.write(f"- Total Records: {len(df)}\n")
        f.write(f"- Attack Percentage: {(df['attack'].sum() / len(df)) * 100:.2f}%\n\n")
        
        if dt_model:
            dt_pred = dt_model.predict(X_test)
            f.write("Decision Tree Results:\n")
            f.write(f"- Accuracy: {accuracy_score(y_test, dt_pred):.4f}\n")
            f.write(f"- Precision: {precision_score(y_test, dt_pred, average='weighted'):.4f}\n")
            f.write(f"- Recall: {recall_score(y_test, dt_pred, average='weighted'):.4f}\n")
            f.write(f"- F1 Score: {f1_score(y_test, dt_pred, average='weighted'):.4f}\n\n")
        
        if rf_model:
            rf_pred = rf_model.predict(X_test)
            f.write("Random Forest Results:\n")
            f.write(f"- Accuracy: {accuracy_score(y_test, rf_pred):.4f}\n")
            f.write(f"- Precision: {precision_score(y_test, rf_pred, average='weighted'):.4f}\n")
            f.write(f"- Recall: {recall_score(y_test, rf_pred, average='weighted'):.4f}\n")
            f.write(f"- F1 Score: {f1_score(y_test, rf_pred, average='weighted'):.4f}\n\n")

def run_pipeline():
    # argparse is a standard Python library used to parse command-line arguments.
    # It automatically generates help messages and ensures that the user provides
    # the required inputs in the correct format. CLI interfaces are essential for
    # security tools because they allow for automation, integration into shell 
    # scripts, and remote execution on servers where graphical interfaces are 
    # unavailable or impractical.
    parser = argparse.ArgumentParser(description="Intrusion Detection System Pipeline")
    parser.add_argument("--data", required=True, help="Path to the dataset file")
    parser.add_argument("--model", choices=['dt', 'rf', 'both'], default='both', help="Model(s) to train")
    parser.add_argument("--report", action='store_true', help="Generate report.txt")
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

    print("||||||||| Intrusion Detection System |||||||||")
    
    print("[1/6] Loading dataset...")
    df = pd.read_csv(args.data, names=columns, header=None)
    
    print("[2/6] Preprocessing data...")
    # Correct order: simplify labels first, then encode features
    df = preprocessing.simplify_labels(df)
    df = preprocessing.encode_features(df)
    df, scaler = preprocessing.scale_features(df)
    X_train, X_test, y_train, y_test = preprocessing.split_data(df)
    
    dt_model = None
    rf_model = None
    
    if args.model in ['dt', 'both']:
        print("[3/6] Training Decision Tree...")
        dt_model = model.train_decision_tree(X_train, y_train)
        model.evaluate_model(dt_model, X_test, y_test)
        model.plot_confusion_matrix(dt_model, X_test, y_test)
    
    if args.model in ['rf', 'both']:
        print("[4/6] Training Random Forest...")
        rf_model = model.train_random_forest(X_train, y_train)
        model.evaluate_model(rf_model, X_test, y_test)
        model.plot_confusion_matrix(rf_model, X_test, y_test)
    
    print("[5/6] Running anomaly detection...")
    clustering.detect_anomalies(X_train, y_train)
    
    if args.report:
        print("[6/6] Generating report...")
        generate_report(df, dt_model, rf_model, X_test, y_test)
    
    print("Pipeline complete.")

if __name__ == "__main__":
    run_pipeline()
