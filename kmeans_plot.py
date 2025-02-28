import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import gunicorn

def generate_kmeans_plot():
    df = pd.read_csv("Mall_Customers.csv")
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    X = df[features]
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)
    plt.figure(figsize=(8, 5))
    sns.heatmap(kmeans.cluster_centers_, annot=True, cmap='coolwarm', xticklabels=features)
    plt.xlabel("Features")
    plt.ylabel("Cluster")
    plt.title("Cluster Heatmap: K-Means on Mall Customers")
    plt.savefig("static/kmeans_heatmap.png")
    plt.close()

