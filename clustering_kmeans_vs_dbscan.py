import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Clustering Visualization", layout="wide")

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("Clustering Parameters")

# KMeans parameters
k = st.sidebar.slider("Number of Clusters (KMeans)", min_value=2, max_value=10, value=5, step=1)

# DBSCAN parameters
eps = st.sidebar.slider("EPS (DBSCAN)", min_value=0.1, max_value=3.0, value=0.5, step=0.1)
min_samples = st.sidebar.slider("min_samples (DBSCAN)", min_value=1, max_value=20, value=5, step=1)

# Data Selection
st.sidebar.header("Data Settings")
datafile = st.sidebar.text_input("Data file path", "players.csv")

features_input = st.sidebar.text_input("Features (comma-separated)", "minutes,goals_scored,assists,influence,threat,creativity")
features = [f.strip() for f in features_input.split(",") if f.strip() != ""]

st.sidebar.markdown("Click 'Run Clustering' after changing parameters.")

run_clustering = st.sidebar.button("Run Clustering")


@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

if run_clustering:
    try:
        df = load_data(datafile)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()


    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        st.error(f"The following features are not found in the dataset: {missing_features}")
        st.stop()

    df_filtered = df[features].dropna()
    X = df_filtered.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans Clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    df_filtered['kmeans_cluster'] = kmeans_labels
    kmeans_counts = df_filtered['kmeans_cluster'].value_counts()

    kmeans_sil_score = silhouette_score(X_scaled, kmeans_labels) if len(set(kmeans_labels)) > 1 else np.nan


    # DBSCAN 
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    df_filtered['dbscan_cluster'] = dbscan_labels
    dbscan_counts = df_filtered['dbscan_cluster'].value_counts()

    # Calculate DBSCAN silhouette if possible
    unique_dbscan_labels = set(dbscan_labels)
    if len(unique_dbscan_labels) > 1 and -1 not in unique_dbscan_labels:
        dbscan_sil_score = silhouette_score(X_scaled, dbscan_labels)
    else:
        dbscan_sil_score = np.nan

    # PCA for Visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=kmeans_labels, palette='Set1', ax=axes[0])
    axes[0].set_title('KMeans Clusters')

    dbscan_palette = sns.color_palette("Set2", n_colors=len(unique_dbscan_labels))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=dbscan_labels, palette=dbscan_palette, ax=axes[1])
    axes[1].set_title('DBSCAN Clusters')

    plt.tight_layout()

    st.subheader("Clustering Results")

    col1, col2 = st.columns(2)
    with col1:
        st.write("### KMeans Results")
        st.write("**Cluster Counts:**")
        st.dataframe(kmeans_counts.rename("Count"))
        st.write(f"**Silhouette Score:** {kmeans_sil_score:.4f}" if not np.isnan(kmeans_sil_score) else "N/A")

    with col2:
        st.write("### DBSCAN Results")
        st.write("**Cluster Counts:**")
        st.dataframe(dbscan_counts.rename("Count"))
        st.write(f"**Silhouette Score:** {dbscan_sil_score:.4f}" if not np.isnan(dbscan_sil_score) else "N/A")

    st.pyplot(fig)

    st.markdown("""
    - **KMeans:** Partitions the data into a predefined number (k) of clusters, each oriented around a centroid. The clusters are often spherical when viewed in reduced dimensions.
    - **DBSCAN:** Groups together points that are closely packed, marking outliers as noise (`-1`). Clusters can take irregular shapes. Useful for identifying core samples of high density and distinguishing them from outliers or noise.
    """)
else:
    st.title("Clustering Visualization using KMeans and DBSCAN")
    st.write("Use the sidebar to load data, set parameters, and then click 'Run Clustering' to view the results.")
