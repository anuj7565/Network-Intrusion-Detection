import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_features(df):
    """
    Encodes categorical features in the DataFrame using LabelEncoder.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with specified categorical columns encoded.
    """
    # Encoding converts categorical labels into numerical format.
    # This is necessary because most machine learning algorithms require numerical input
    # and cannot directly process string-based categorical data.
    # 'protocol_type', 'service', and 'flag' are categorical as they represent distinct,
    # non-ordered categories (e.g., 'tcp', 'udp', 'http', 'ftp').
    categorical_cols = ["protocol_type", "service", "flag"]
    
    # Initialize LabelEncoder
    le = LabelEncoder()

    for col in categorical_cols:
        print(f"--- Encoding column: {col} ---")
        # Print unique values before encoding
        print(f"Unique values before encoding for '{col}':\n{df[col].unique()}\n")

        # Apply Label Encoding
        df[col] = le.fit_transform(df[col])

        # Print unique values after encoding
        print(f"Unique values after encoding for '{col}':\n{df[col].unique()}\n")
        
    return df

def simplify_labels(df):
    """
    Simplifies the 'label' column into a binary 'attack' column (0 for normal, 1 for attack).

    Args:
        df (pd.DataFrame): The input DataFrame with a 'label' column.

    Returns:
        pd.DataFrame: The DataFrame with the simplified 'attack' column and 'label' column dropped.
    """
    # Simplifying the labels to binary classification (normal vs. attack) is often
    # a first step in intrusion detection systems. It reduces the complexity of the
    # classification problem from multi-class to binary, which can make initial
    # model training and evaluation simpler and more interpretable.
    # It also helps in identifying overall malicious activity before diving into
    # specific attack types.
    print("--- Simplifying labels to binary classification ---")
    # Create a new column 'attack' where 'normal.' is 0 and all other labels are 1.
    df['attack'] = df['label'].apply(lambda x: 0 if x == 'normal.' else 1)

    # Print the counts of 0s and 1s in the new 'attack' column.
    print("Count of 0s (normal) and 1s (attack) in 'attack' column:")
    print(df['attack'].value_counts())
    print("\n")

    # Drop the original 'label' column as it's no longer needed after creating 'attack'.
    df = df.drop('label', axis=1)
    
    return df
