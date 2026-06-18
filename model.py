from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def train_decision_tree(X_train, y_train):
    """
    Trains a Decision Tree Classifier model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.

    Returns:
        sklearn.tree.DecisionTreeClassifier: The trained Decision Tree model.
    """
    # max_depth controls the maximum depth of the decision tree.
    # Limiting the max_depth helps to prevent the model from overfitting to the
    # training data by creating a tree that is too complex. A deeper tree can
    # capture more specific patterns in the training data, but it might fail
    # to generalize well to new, unseen data. Setting a maximum depth helps
    # to find a balance between bias and variance.
    print("--- Training Decision Tree Classifier on X_train/y_train ---")
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    print("Decision Tree Classifier trained successfully.\n")
    return model

def train_random_forest(X_train, y_train):
    """
    Trains a Random Forest Classifier model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.

    Returns:
        sklearn.ensemble.RandomForestClassifier: The trained Random Forest model.
    """
    # n_estimators defines the number of trees in the forest. Increasing this 
    # generally improves performance and stability but increases computation time.
    # Random Forest is typically more accurate than a single Decision Tree because 
    # it uses an ensemble approach (bagging), which reduces variance and helps 
    # prevent overfitting by averaging the predictions of multiple diverse trees.
    print("--- Training Random Forest Classifier on X_train/y_train ---")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    print("Random Forest Classifier trained successfully.\n")
    
    # Plot feature importance
    importances = model.feature_importances_
    feature_names = X_train.columns
    feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False).head(15)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_imp_df)
    plt.title('Top 15 Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model performance using classification metrics on the test set.

    Metrics in the context of Intrusion Detection:
    - Accuracy: Overall percentage of correctly classified traffic (normal vs attack).
    - Precision: Of all traffic flagged as an attack, how many were actually attacks? 
      High precision reduces false alarms.
    - Recall (Sensitivity): Of all actual attacks, how many did we catch? 
      Crucial for security: A False Negative (missing an attack) means a security 
      breach goes undetected, which is often the most dangerous outcome.
    - F1 Score: The harmonic mean of Precision and Recall, providing a balance 
      between the two.
    - Confusion Matrix: Shows TP (True Positives), TN (True Negatives), 
      FP (False Positives), and FN (False Negatives).
    """
    print("--- Evaluating model on X_test/y_test ---")
    y_pred = model.predict(X_test)
    
    print("--- Model Evaluation ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred, average='weighted'):.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nFalse Positives (FP): {fp}")
    print("Note: A False Positive is a 'False Alarm'—normal traffic incorrectly flagged as an attack. This can lead to 'alert fatigue' for security analysts.")
    
    print(f"False Negatives (FN): {fn}")
    print("Note: A False Negative is a 'Missed Attack'—malicious traffic incorrectly labeled as normal. This is the most dangerous outcome in security as it allows a breach to go undetected.")
    
    print("\nConfusion Matrix:")
    print(f"TN: {tn} | FP: {fp}")
    print(f"FN: {fn} | TP: {tp}")
    print("\n")

def plot_confusion_matrix(model, X_test, y_test):
    """
    Creates and saves a heatmap of the confusion matrix.

    How to read a confusion matrix:
    - The rows represent the Actual classes (Normal vs Attack).
    - The columns represent the Predicted classes (Normal vs Attack).
    - Diagonal elements (top-left to bottom-right) represent correct predictions.
    - Off-diagonal elements represent misclassifications (errors).
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Attack'], 
                yticklabels=['Normal', 'Attack'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
