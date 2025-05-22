import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tensorflow.keras.datasets import fashion_mnist
from PIL import Image
import io

@st.cache_data
def load_data():
    (train_images, _), _ = fashion_mnist.load_data()
    train_images = train_images / 255.0
    train_flat = train_images.reshape(len(train_images), -1)
    return train_images, train_flat

@st.cache_data
def fit_models(train_flat):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(train_flat)
    kmeans = KMeans(n_clusters=10, random_state=42)
    labels = kmeans.fit_predict(train_flat)
    return pca, kmeans, reduced, labels

train_images, train_flat = load_data()
pca, kmeans, train_2d, cluster_labels = fit_models(train_flat)

st.title("Fashion MNIST Clustering Explorer")

option = st.radio("Choose Input Method:", ["Upload Image", "Select Sample Image"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload a 28x28 grayscale image (.png)", type=["png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L").resize((28, 28))
        user_image = np.array(image) / 255.0
else:
    index = st.slider("Select image index from dataset", 0, len(train_images)-1, 0)
    user_image = train_images[index]

if 'user_image' in locals():
    st.image(user_image, caption="Selected Image", width=150)
    user_flat = user_image.reshape(1, -1)

    cluster_id = kmeans.predict(user_flat)[0]
    st.markdown(f"### Predicted Cluster: {cluster_id}")

    dists = np.linalg.norm(train_flat - user_flat, axis=1)
    top_idx = np.argmin(dists)
    st.image(train_images[top_idx], caption="Most Similar Item", width=150)

    user_2d = pca.transform(user_flat)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(train_2d[:, 0], train_2d[:, 1], c=cluster_labels, cmap='tab10', alpha=0.3, s=10)
    ax.scatter(user_2d[0, 0], user_2d[0, 1], color='red', edgecolor='black', s=100, label="Your Image")
    ax.legend()
    ax.set_title("PCA Projection with User Image Highlighted")
    st.pyplot(fig)
