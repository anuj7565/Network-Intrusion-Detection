import pandas as pd
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
    
    # Calculate metrics
    dt_pred = dt_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)
    
    total_records = len(df)
    attack_pct = (df['attack'].sum() / total_records) * 100
    
    # Get top 5 features
    importances = rf_model.feature_importances_
    feature_names = X_test.columns
    feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    top_5 = feature_imp_df.sort_values(by='Importance', ascending=False).head(5)
    
    with open('report.txt', 'w') as f:
        f.write(f"Intrusion Detection System Report\n")
        f.write(f"Generated on: {datetime.datetime.now()}\n\n")
        f.write(f"Dataset Statistics:\n")
        f.write(f"- Total Records: {total_records}\n")
        f.write(f"- Attack Percentage: {attack_pct:.2f}%\n\n")
        
        f.write("Decision Tree Results:\n")
        f.write(f"- Accuracy: {accuracy_score(y_test, dt_pred):.4f}\n")
        f.write(f"- Precision: {precision_score(y_test, dt_pred, average='weighted'):.4f}\n")
        f.write(f"- Recall: {recall_score(y_test, dt_pred, average='weighted'):.4f}\n")
        f.write(f"- F1 Score: {f1_score(y_test, dt_pred, average='weighted'):.4f}\n\n")
        
        f.write("Random Forest Results:\n")
        f.write(f"- Accuracy: {accuracy_score(y_test, rf_pred):.4f}\n")
        f.write(f"- Precision: {precision_score(y_test, rf_pred, average='weighted'):.4f}\n")
        f.write(f"- Recall: {recall_score(y_test, rf_pred, average='weighted'):.4f}\n")
        f.write(f"- F1 Score: {f1_score(y_test, rf_pred, average='weighted'):.4f}\n\n")
        
        f.write("Top 5 Important Features (Random Forest):\n")
        for _, row in top_5.iterrows():
            f.write(f"- {row['Feature']}: {row['Importance']:.4f}\n")
            
        f.write("\nConclusion:\n")
        if rf_model.score(X_test, y_test) > dt_model.score(X_test, y_test):
            f.write("The Random Forest model outperformed the Decision Tree, likely due to its ensemble nature reducing variance.")
        else:
            f.write("The Decision Tree performed comparably to the Random Forest.")

def run_pipeline():
    # Printing progress updates is crucial in a data pipeline because these 
    # processes (especially training and scaling) can be time-consuming. 
    # Progress updates provide visibility into the system's state, help 
    # identify where a failure might occur, and provide an estimated 
    # time to completion for the user.
    
    print("||||||||| Intrusion Detection System |||||||||")
    
    # [1/6] Loading dataset
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
    generate_report(df, dt_model, rf_model, X_test, y_test)
    
    print("\n--- Final Summary ---")
    print(f"Decision Tree Recall: {dt_recall:.4f}")
    print(f"Random Forest Recall: {rf_recall:.4f}")
    print("Pipeline complete. Check generated images and report.txt for details.")

if __name__ == "__main__":
    run_pipeline()
