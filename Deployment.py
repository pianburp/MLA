import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt

# Load and preprocess data
@st.cache
def load_data():
    data = pd.read_csv('data.csv', encoding='latin1')
    data_cleaned = data[(data['Quantity'] > 0) & (data['UnitPrice'] > 0)]
    data_cleaned['CustomerID'].fillna('Unknown', inplace=True)
    return data_cleaned

data_cleaned = load_data()

# Sidebar - Clustering options
st.sidebar.title('Clustering Options')
algorithm = st.sidebar.selectbox('Select Algorithm', ['K-Means', 'DBSCAN'])

if algorithm == 'K-Means':
    n_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=10, value=3)
else:
    eps = st.sidebar.slider('Epsilon (eps)', min_value=0.1, max_value=2.0, value=0.5)
    min_samples = st.sidebar.slider('Minimum Samples', min_value=5, max_value=100, value=5)

# Clustering
def cluster_data(method, **kwargs):
    features = data_cleaned[['Quantity', 'UnitPrice']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)

    if method == 'K-Means':
        model = KMeans(n_clusters=kwargs['n_clusters'], random_state=42)
    elif method == 'DBSCAN':
        model = DBSCAN(eps=kwargs['eps'], min_samples=kwargs['min_samples'])

    data_cleaned['Cluster'] = model.fit_predict(features_pca)
    return features_pca, data_cleaned['Cluster']

# Plotting
def plot_clusters(features_pca, clusters):
    plt.figure(figsize=(10, 6))
    plt.scatter(features_pca[:, 0], features_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    plt.title(f'{algorithm} Clustering Results')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    st.pyplot(plt)

# Main
if st.sidebar.button('Cluster'):
    if algorithm == 'K-Means':
        features_pca, clusters = cluster_data('K-Means', n_clusters=n_clusters)
    else:
        features_pca, clusters = cluster_data('DBSCAN', eps=eps, min_samples=min_samples)
    plot_clusters(features_pca, clusters)

st.title('Clustering Application')
st.write('This application performs clustering on the provided dataset using selected parameters.')
