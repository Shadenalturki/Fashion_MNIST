# 🧠 Fashion MNIST Clustering Explorer

A Streamlit web app that visualizes Fashion MNIST image clusters using PCA and KMeans. Upload a 28x28 grayscale image or choose one from the dataset to see which cluster it belongs to — and find the most similar item!

## 🚀 Features

- Upload your own 28x28 grayscale `.png` image or pick one from the dataset
- Predicts which cluster the image belongs to using KMeans
- Visualizes clusters with PCA in 2D space
- Highlights your image on the cluster map
- Shows the most similar image from the dataset
- Cluster labels are annotated on the PCA plot

## 📦 Dependencies

Make sure you have the following Python packages installed:

```bash
streamlit
numpy
matplotlib
scikit-learn
tensorflow
Pillow
