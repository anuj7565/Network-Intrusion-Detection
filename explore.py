import pandas as pd

# Section: Load Dataset
# This section loads the 'kddcup.data_10_percent.gz' dataset using pandas.
# It specifies the column names as provided, ensuring proper parsing of the data
# and handling of the gzipped file format.
column_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
]

try:
    df = pd.read_csv(
        'kddcup.data_10_percent.gz',
        compression='gzip',
        names=column_names
    )
    print("Dataset loaded successfully.\n")
except FileNotFoundError:
    print("Error: 'kddcup.data_10_percent.gz' not found.")
    print("Please ensure the dataset file is in the same directory as the script.")
    exit()

# Section: Print Dataset Shape
# This section prints the number of rows and columns in the loaded dataset.
# Understanding the shape helps to quickly grasp the size of the data being analyzed.
print("Dataset Shape:")
print(df.shape)
print("\n")

# Section: Print Column Names
# This section iterates through and prints all the column names present in the DataFrame.
# This is useful for verifying that columns were loaded correctly and for reference
# during data exploration and manipulation.
print("Column Names:")
for col in df.columns:
    print(col)
print("\n")

# Section: Print Count of Each Attack Type
# This section calculates and prints the unique counts of each value in the 'label' column.
# The 'label' column typically indicates different attack types or 'normal' connections.
# This provides an immediate overview of the distribution of different classes in the dataset.
print("Count of each attack type in 'label' column:")
print(df['label'].value_counts())
print("\n")

# Section: Print Normal vs. Attack Records Count
# This section categorizes each record as either 'normal' or 'attack' based on the 'label' column.
# It then counts and prints the total number of 'normal' records versus 'attack' records.
# This gives a high-level view of the balance between benign and malicious traffic,
# which is critical for understanding the dataset's characteristics in intrusion detection.
print("Normal vs. Attack records count:")
df['attack_type'] = df['label'].apply(lambda x: 'normal' if x == 'normal.' else 'attack')
print(df['attack_type'].value_counts())
print("\n")
