from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib

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
    joblib.dump(model, 'rf_model.pkl')
    print("Saved trained model to rf_model.pkl")
    
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

def cross_validate_model(model, X, y, model_name):
    """
    Performs k-fold cross-validation to assess model stability.

    Cross-validation involves splitting the dataset into 'k' subsets (folds). 
    The model is trained on k-1 folds and validated on the remaining fold, 
    repeating this process until every fold has been used as a test set. 
    This provides a more robust estimate of model performance than a single 
    train-test split.

    The standard deviation of the scores is critical for reliability: a low 
    standard deviation indicates that the model's performance is consistent 
    across different subsets of data, whereas a high standard deviation 
    suggests the model is sensitive to the specific data it is trained on.
    """
    print(f"--- Cross-Validating {model_name} ---")
    results = {}
    metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    
    for metric in metrics:
        scores = cross_val_score(model, X, y, cv=5, scoring=metric)
        results[metric] = scores
        print(f"{metric.replace('_weighted', '').capitalize()}: {scores.mean():.4f} +/- {scores.std():.4f}")
    
    return results

def plot_roc_curve(dt_model, rf_model, X_test, y_test):
    """
    Plots the ROC curves for both models.

    AUC (Area Under the Curve) represents the model's ability to distinguish 
    between classes. An AUC of 1.0 indicates perfect classification, while 
    0.5 represents a model that performs no better than random guessing.

    The diagonal dotted line represents the 'random classifier' baseline. 
    Any model curve falling below this line is performing worse than random 
    chance, while curves above it indicate predictive power.
    """
    plt.figure(figsize=(8, 6))
    
    for model, name in [(dt_model, "Decision Tree"), (rf_model, "Random Forest")]:
        y_probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Baseline')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig('roc_curve.png')
    plt.close()
    
    # Return AUCs for reporting
    dt_probs = dt_model.predict_proba(X_test)[:, 1]
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    return auc(roc_curve(y_test, dt_probs)[0], roc_curve(y_test, dt_probs)[1]), auc(roc_curve(y_test, rf_probs)[0], roc_curve(y_test, rf_probs)[1])

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
    print("Confusion matrix saved to confusion_matrix.png")
    plt.close()
