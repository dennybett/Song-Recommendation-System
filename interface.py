import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Load ML model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# Future use
kmeans_model = joblib.load("output/kmeans_model.pkl")
agglo_model = joblib.load("output/agglo_model.pkl")
gaussian_model = joblib.load("output/gaussian_model.pkl")

# Load trained model for K-Means without scaling
kmeans_nonscaled_model = joblib.load("output/kmeans_nonscale_model.pkl")

# Future use
# Import and read output from the main notebook
cleaned_data = pd.read_csv("output/df_cleaned_with_labels.csv")
scaled_data = pd.read_csv("output/df_scaled_with_labels.csv")

# Import and read output from the main notebook for the non-scaled data
cleaned2_data = pd.read_csv("output/df_cleaned2_with_labels.csv")
non_scaled_data = pd.read_csv("output/df_non_scaled_with_labels.csv")

# Max_Records
max_records = 10

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
def find_top_match(model, df, method, features):
    # Find all songs in the same cluster using the cleaned data
    if method == "K-Means":
        prediction = model.predict(features)
        cluster_songs = df[df["kmeans_labels"] == prediction[0]]

        # Debug Code
        st.success(f'Cluster: {prediction[0]}')
        #st.success(len(cluster_songs))
    
    # Export the matched songs to a CSV file for testing
    cluster_songs.to_csv('output/cluster_songs_matched.csv')

    # Select random song(s) from the cluster
    top_match = cluster_songs.sample(max_records)[["track_name", "artist"]].values.tolist()

    # Debug Code
    #st.success(len(top_match))

    return top_match

# Recommend track using K-Means Clustering
if st.button("Find My Song (K-Means) 🎧"):
    top_match = find_top_match(kmeans_nonscaled_model, cleaned2_data, "K-Means", features)

    # Debug Code
    #st.success(top_match)

    st.success(f":musical_note: Best {max_records} matches for your input:")
    for i in range(0, max_records):
        st.write(f"\n:musical_score: Track Name: **{top_match[i][0]}**  \n:male-singer: Artist Name: **{top_match[i][1]}**")
