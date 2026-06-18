from sklearn.tree import DecisionTreeClassifier

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
