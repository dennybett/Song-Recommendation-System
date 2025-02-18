import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Load ML model
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Load trained model
model = joblib.load("output/kmeans_model.pkl")

# Import and read output from the main notebook
cleaned_data = pd.read_csv("output/df_cleaned_with_labels.csv")
scaled_data = pd.read_csv("output/df_scaled_with_labels.csv")

# Streamlit UI
st.title(":musical_note: Spotify Song Recommendation")
st.write("Enter song characteristics to get recommended tracks!")

# Sidebar sliders for inputs
st.sidebar.header("Input Features")
danceability = st.sidebar.slider("Danceability (Low Danceability <-> High Danceability)", 0.0, 1.0, 0.5)
energy = st.sidebar.slider("Energy (Low Energy <-> High Energy)", 0.0, 1.0, 0.5)
speechiness = st.sidebar.slider("Speechiness (Low Presence of Words <-> High Presence of Words)", 0.0, 1.0, 0.1)
acousticness = st.sidebar.slider("Acousticness (Electronic <-> Acoustic)", 0.0, 1.0, 0.3)
instrumentalness = st.sidebar.slider("Instrumentalness (Low Presence of Instruments <-> High Presence of Instruments)", 0.0, 1.0, 0.0)
liveness = st.sidebar.slider("Liveness (Live <-> Studio)", 0.0, 1.0, 0.2)
valence = st.sidebar.slider("Mood (Sad <-> Happy)", 0.0, 1.0, 0.5)
tempo = st.sidebar.slider("Tempo (BPM) (Slow <-> Fast)", 50.0, 200.0, 120.0)

# Convert inputs to array
features = np.array([[danceability, energy, speechiness, acousticness, instrumentalness, liveness, valence, tempo]])

# Function to find best match
def find_best_match(model, df, method, features):
    # Get the cluster label for the input features
    prediction = model.predict(features)
    
    # Find all songs in the same cluster using the cleaned data
    if method == "K-Means":
        cluster_songs = df[df["kmeans_labels"] == prediction[0]]
    elif method == "Agglomerative":
        cluster_songs = df[df["agglo_labels"] == prediction[0]]
    elif method == "Gaussian":
        cluster_songs = df[df["gaussian_labels"] == prediction[0]]
    
    # Export the matched songs to a CSV file for testing
    cluster_songs.to_csv('output/cluster_songs_matched.csv')

    # Select a random song from the cluster
    best_match = cluster_songs.sample(1)[["track_name", "artist"]].values[0]
    return best_match

# Recommend track using K-Means Clustering
if st.button("Find My Song (K-Means) 🎧"):
    best_match = find_best_match(model, cleaned_data, "K-Means", features)
    st.success(f":musical_note: Best match for your input:  \n:microphone: Track Name: **{best_match[0]}**  \n:female-singer: Artist Name: **{best_match[1]}**")

# Recommend track using Agglomerative Clustering
if st.button("Find My Song (Agglomerative) 🎧"):
    best_match = find_best_match(model, cleaned_data, "Agglomerative", features)
    st.success(f":musical_note: Best match for your input:  \n:microphone: Track Name: **{best_match[0]}**  \n:female-singer: Artist Name: **{best_match[1]}**")

# Recommend track using Gaussian Clustering
if st.button("Find My Song (Gaussian) 🎧"):
    best_match = find_best_match(model, cleaned_data, "Gaussian", features) 
    st.success(f":musical_note: Best match for your input:  \n:microphone: Track Name: **{best_match[0]}**  \n:female-singer: Artist Name: **{best_match[1]}**")
