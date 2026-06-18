import pandas as pd
from sklearn.metrics import recall_score
import preprocessing
import model
import clustering

def run_pipeline():
    # Printing progress updates is crucial in a data pipeline because these 
    # processes (especially training and scaling) can be time-consuming. 
    # Progress updates provide visibility into the system's state, help 
    # identify where a failure might occur, and provide an estimated 
    # time to completion for the user.
    
    print("||||||||| Intrusion Detection System |||||||||")
    
    # [1/6] Loading dataset
    # Assuming a CSV file named 'network_data.csv' exists
    print("[1/6] Loading dataset...")
    df = pd.read_csv('network_data.csv')
    
    # [2/6] Preprocessing data
    print("[2/6] Preprocessing data...")
    df = preprocessing.simplify_labels(df)
    df = preprocessing.encode_features(df)
    df, scaler = preprocessing.scale_features(df)
    X_train, X_test, y_train, y_test = preprocessing.split_data(df)
    
    # [3/6] Training Decision Tree
    print("[3/6] Training Decision Tree...")
    dt_model = model.train_decision_tree(X_train, y_train)
    dt_recall = recall_score(y_test, dt_model.predict(X_test), average='weighted')
    
    # [4/6] Training Random Forest
    print("[4/6] Training Random Forest...")
    rf_model = model.train_random_forest(X_train, y_train)
    rf_recall = recall_score(y_test, rf_model.predict(X_test), average='weighted')
    
    # [5/6] Running anomaly detection
    print("[5/6] Running anomaly detection...")
    clustering.detect_anomalies(X_train, y_train)
    
    # [6/6] Generating report
    print("[6/6] Generating report...")
    print("\n--- Final Summary ---")
    print(f"Decision Tree Recall: {dt_recall:.4f}")
    print(f"Random Forest Recall: {rf_recall:.4f}")
    print("Pipeline complete. Check generated images for visualizations.")

if __name__ == "__main__":
    run_pipeline()
