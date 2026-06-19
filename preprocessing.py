"""
This module provides functions for preprocessing data, including
encoding categorical features, splitting data into training and testing sets,
scaling numerical features, and simplifying target labels.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def encode_features(df):
    """Encodes categorical features in the DataFrame using LabelEncoder.
    Args:
        df (pd.DataFrame): The input DataFrame containing categorical columns
                          'protocol_type', 'service', and 'flag'.

    Returns:
        pd.DataFrame: The DataFrame with the specified categorical columns encoded
                      into numerical format.
    """
    categorical_cols = ["protocol_type", "service", "flag"]

    le = LabelEncoder()

    for col in categorical_cols:
        if col in df.columns:
            print(f"--- Encoding column: {col} ---")
            print(f"Unique values before encoding for '{col}':\n{df[col].unique()}\n")
            df[col] = le.fit_transform(df[col])
            print(f"Unique values after encoding for '{col}':\n{df[col].unique()}\n")

    return df

def split_data(df):
    """Separates features (X) and target (y), then splits the data into
    training and testing sets.

    Args:
        df (pd.DataFrame): The input DataFrame which must include the 'attack' column.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: X_train (features for training)
            - pd.DataFrame: X_test (features for testing)
            - pd.Series: y_train (target for training)
            - pd.Series: y_test (target for testing)
    """
    print("--- Splitting data into training and testing sets ---")

    X = df.drop('attack', axis=1)
    y = df['attack']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of y_test: {y_test.shape}")
    print("\n")

    return X_train, X_test, y_train, y_test

def scale_features(df):
    """Scales numerical features using StandardScaler.

    StandardScaler transforms features to have a mean of 0 and a standard deviation of 1.
    This is crucial for many machine learning algorithms that are sensitive to feature scales.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The DataFrame with numerical features scaled.
            - sklearn.preprocessing.StandardScaler: The fitted StandardScaler object.
    """
    print("--- Scaling numerical features using StandardScaler ---")

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if 'attack' in numeric_cols:
        numeric_cols.remove('attack')

    print(f"Before scaling 'src_bytes': Mean = {df['src_bytes'].mean():.2f}, Std Dev = {df['src_bytes'].std():.2f}")

    scaler = StandardScaler()

    df_scaled_features = pd.DataFrame(scaler.fit_transform(df[numeric_cols]),
                                      columns=numeric_cols,
                                      index=df.index)

    df_scaled = df_scaled_features
    if 'attack' in df.columns:
        df_scaled['attack'] = df['attack']

    print(f"After scaling 'src_bytes': Mean = {df_scaled['src_bytes'].mean():.2f}, Std Dev = {df_scaled['src_bytes'].std():.2f}")
    print("\n")

    return df_scaled, scaler

def simplify_labels(df):
    """Simplifies the 'label' column into a binary 'attack' column (0 for normal, 1 for attack).

    This process reduces a multi-class classification problem to a binary one,
    simplifying initial model training and evaluation for intrusion detection.

    Args:
        df (pd.DataFrame): The input DataFrame with a 'label' column.

    Returns:
        pd.DataFrame: The DataFrame with the simplified 'attack' column and 'label' column dropped.

    Raises:
        KeyError: If the 'label' column is not found in the DataFrame.
    """
    print(f"--- Simplifying labels. Current columns: {df.columns.tolist()} ---")

    if 'label' not in df.columns:
        raise KeyError("The 'label' column is missing from the DataFrame. Ensure simplify_labels is called before dropping it.")

    print("--- Simplifying labels to binary classification ---")
    df['attack'] = df['label'].apply(lambda x: 0 if x == 'normal.' else 1)

    print("Count of 0s (normal) and 1s (attack) in 'attack' column:")
    print(df['attack'].value_counts())
    print("\n")

    df = df.drop('label', axis=1)

    return df
