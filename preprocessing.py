import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

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

def split_data(df):
    """
    Separates features (X) and target (y), then splits the data into
    training and testing sets.

    Args:
        df (pd.DataFrame): The input DataFrame which should include the 'attack' column.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: X_train (features for training)
            - pd.DataFrame: X_test (features for testing)
            - pd.Series: y_train (target for training)
            - pd.Series: y_test (target for testing)
    """
    # Splitting the data into training and testing sets is a fundamental step
    # in machine learning. It allows us to evaluate the performance of our model
    # on unseen data, preventing overfitting. The model learns patterns from the
    # training data and its generalization capability is tested on the test data.
    print("--- Splitting data into training and testing sets ---")
    
    # Separate features (X) and target (y)
    X = df.drop('attack', axis=1)
    y = df['attack']

    # Split the data into 80% train and 20% test.
    # random_state ensures that the split is reproducible. If you run the code
    # multiple times with the same random_state, you will get the same train/test split.
    # This is crucial for consistent experimentation and debugging.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Print the shapes of the resulting datasets
    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of y_test: {y_test.shape}")
    print("\n")

    return X_train, X_test, y_train, y_test

def scale_features(df):
    """
    Scales numerical features using StandardScaler.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The DataFrame with numerical features scaled.
            - sklearn.preprocessing.StandardScaler: The fitted StandardScaler object.
    """
    # StandardScaler transforms features to have a mean of 0 and a standard deviation of 1.
    # Mathematically, for each feature x, the scaled value x_scaled is calculated as:
    # x_scaled = (x - mean(x)) / std(x)
    # This process is crucial for many machine learning algorithms (e.g., K-Means, SVMs, Neural Networks)
    # because they are sensitive to the scale of input features. Features with larger ranges
    # can dominate the distance calculations or weight updates, leading to suboptimal model performance.
    # Scaling ensures that all features contribute equally to the model's learning process.
    print("--- Scaling numerical features using StandardScaler ---")
    
    # Separate the target variable 'attack' if it exists, otherwise get all numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if 'attack' in numeric_cols:
        numeric_cols.remove('attack')
    
    # Print mean and standard deviation of 'src_bytes' before scaling
    print(f"Before scaling 'src_bytes': Mean = {df['src_bytes'].mean():.2f}, Std Dev = {df['src_bytes'].std():.2f}")

    # Initialize StandardScaler
    scaler = StandardScaler()

    # Apply StandardScaler to the identified numerical columns
    df_scaled_features = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), 
                                      columns=numeric_cols, 
                                      index=df.index)

    # Combine scaled features with the 'attack' column (if it existed) and non-numeric columns
    df_scaled = df_scaled_features
    if 'attack' in df.columns:
        df_scaled['attack'] = df['attack']
    
    # Print mean and standard deviation of 'src_bytes' after scaling
    print(f"After scaling 'src_bytes': Mean = {df_scaled['src_bytes'].mean():.2f}, Std Dev = {df_scaled['src_bytes'].std():.2f}")
    print("\n")

    return df_scaled, scaler

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
