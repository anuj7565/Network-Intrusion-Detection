from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

def detect_anomalies(X, y):
    """
    Performs K-Means clustering to identify potential anomalies.

    Clustering is an unsupervised learning technique that can detect unknown 
    attacks (zero-day exploits) that supervised models might miss. Because 
    supervised models are trained on known attack signatures, they struggle 
    with novel patterns. Clustering, however, groups data based on inherent 
    similarities; if an attack creates a distinct pattern of behavior, it 
    may form its own cluster even if the model has never seen that specific 
    attack before.
    """
    # 1. Use KMeans with n_clusters=2
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    # 2. Assign clusters to a dataframe for analysis
    df_results = pd.DataFrame({'Cluster': clusters, 'Actual': y})
    
    # 3 & 4. Compare cluster assignments with actual labels
    print("--- Clustering Analysis ---")
    for cluster_id in range(2):
        cluster_data = df_results[df_results['Cluster'] == cluster_id]
        attack_count = cluster_data['Actual'].sum()
        print(f"Cluster {cluster_id}: {len(cluster_data)} total records, {attack_count} actual attacks.")
    
    # 5. Save scatter plot using first two features
    plt.figure(figsize=(10, 6))
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap='viridis', alpha=0.5)
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.title('K-Means Clustering (First Two Features)')
    plt.savefig('clusters.png')
    plt.close()
    
    return clusters
