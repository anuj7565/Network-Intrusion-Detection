from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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
    print("--- Training Decision Tree Classifier ---")
    model = DecisionTreeClassifier(max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    print("Decision Tree Classifier trained successfully.\n")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model performance using classification metrics.

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
    y_pred = model.predict(X_test)
    
    print("--- Model Evaluation ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred, average='weighted'):.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"TN: {cm[0,0]} | FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]} | TP: {cm[1,1]}")
    print("\n")
