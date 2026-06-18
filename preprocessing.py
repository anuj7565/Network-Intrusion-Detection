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
