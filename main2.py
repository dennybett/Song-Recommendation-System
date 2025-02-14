# %%
# Import Dependencies
import os
import numpy as np
import pandas as pd
import utils as utils
import seaborn as sns
import plotly.express as px 
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings("ignore")

# %%
# Import and read the data
df_raw = pd.read_csv("data/top_10000_1960-now.csv")
df_raw.head()

# %%
# list columns for features and target
df_raw.columns

# %%
# Drop unnecessary columns
# all columns listed, columns to keep are commented out.
df_data = df_raw.drop([#'Track URI',
                       'Track Name',
                       'Artist URI(s)',
                       'Artist Name(s)',
                       'Album URI',
                       'Album Name',
                       'Album Artist URI(s)',
                       'Album Artist Name(s)',
                       'Album Release Date',
                       'Album Image URL',
                       'Disc Number',
                       'Track Number',
                       'Track Duration (ms)',
                       'Track Preview URL',
                       #'Explicit',
                       'Popularity',
                       'ISRC',
                       'Added By',
                       'Added At',
                       #'Artist Genres',
                       #'Danceability',
                       #'Energy',
                       #'Key',
                       'Loudness',
                       'Mode',
                       #'Speechiness',
                       #'Acousticness',
                       #'Instrumentalness',
                       #'Liveness',
                       #'Valence',
                       #'Tempo',
                       #'Time Signature',
                       'Album Genres',
                       'Label',
                       'Copyrights'],
                       axis=1)

# %%
# Review remaining column names
df_data.columns 

# %%
# Columns renamed to follow convention
df_data = df_data.rename(columns={
                   'Track URI': 'track_uri',
                   'Album Image URL': 'image',
                   'Explicit': 'explicit',
                   'Popularity': 'popularity',
                   'Artist Genres': 'artist_genres',
                   'Danceability': 'danceability',
                   'Energy': 'energy',
                   'Key': 'key',
                   'Speechiness': 'speechiness',
                   'Acousticness': 'acousticness',
                   'Instrumentalness': 'instrumentalness',
                   'Liveness': 'liveness',
                   'Valence': 'valence',
                   'Tempo': 'tempo',
                   'Time Signature': 'time_signature'
       })


# %%
# Verify Update
df_data.columns

# %%
# Dropping null columns
df_data = df_data.dropna(how="any")

# %%
# Reset index on dataframe
df_data = df_data.reset_index(drop=True)


# %%
# utils.plot_correlation_heatmap(df_data)

# %%
# utils.plot_numeric_distributions(df_data)

# %% [markdown]
# #### Cleaning and encoding the ['Artist Genres'] column

# %%
## Cleaning and encoding the 'artist genres' column
# Explore the values
df_data['artist_genres'].value_counts()

# %%
## Cleaning and encoding the 'artist genres' column
# how many unique genre combos are there?
# Explore the values
df_data['artist_genres'].nunique()

# %%
## Cleaning and encoding the 'artist genres' column
# Add a space after any commas if one is not already present
df_data['artist_genres'] = df_data['artist_genres'].str.replace(
    r',(?=\S)', ', ', regex=True
    )


# %%
## Cleaning and encoding the 'artist genres' column
# Verify spaces added
df_data['artist_genres'].value_counts()

# %%
## Cleaning and encoding the 'artist genres' column
# replace spaces with and underscore where a letter character is on either side
df_data['artist_genres'] = df_data['artist_genres'].str.replace(
    r'(?<=[a-zA-Z]) (?=[a-zA-Z])', '_', regex=True
    )

# %%
## Cleaning and encoding the 'artist genres' column
# Verify underscores inserted
df_data['artist_genres'].value_counts()

# %%
## Cleaning and encoding the 'artist genres' column
# Split the ['artist_genres'] stings into lists
df_data['artist_genres'] = df_data['artist_genres'].str.split(', ')

# %%
df_data.head()

# %%
## Cleaning and encoding the 'artist genres' column
# Initialize MultiLabelBinarizer
mlb = MultiLabelBinarizer()

# %%
## Cleaning and encoding the 'artist genres' column
# fit and transform 'Artist Genres' column
encoded_genres = mlb.fit_transform(df_data['artist_genres'])
df_encoded_genres = pd.DataFrame(encoded_genres)
df_encoded_genres.head()

# %%
## Cleaning and encoding the 'artist genres' column
# concatenate back into the original DataFrame
df_encoded = pd.concat([df_data.drop(columns=['artist_genres']), df_encoded_genres], axis=1)

# Handle missing values (if any)
#df_encoded.fillna(0, inplace=True)


# %%
# Encode the ['explicit'] column
df_encoded['explicit'] = df_encoded['explicit'].map({True: 1, False: 0})
df_encoded.head()

# %%
df_encoded.dtypes

# %%
# Create features dataframe
# Set column names as strings
df_x = df_encoded.drop(columns='track_uri')
df_x.columns = df_x.columns.astype(str)

# %%
df_x.head()


# %%
df_data.head()

# %%


# %%
# Running pca without genres column
# # Scale data with Standard Scaler
scaler = StandardScaler()

df_test = df_data.drop(columns=['track_uri', 'artist_genres'])

#scaled_data = scaler.fit_transform(df_test)

# call PCA
pca = PCA(n_components=2)

# fit and apply
genres_pca = pca.fit_transform(df_test)

# Create DataFrame with PCA results
genres_pca_df = pd.DataFrame(
    genres_pca,
    columns=['genre_pca_1',
            'genre_pca_2'
            ])



# %%
pca.explained_variance_ratio_

# %%
# create pca dataframe
pca_test_df = pd.DataFrame()

# %%
# determine k value


# %%
# elbow

# %%
df_data.drop(columns=['artist_genres','time_signature'])

# %%
# ChattGPT reference code#####################################

#KNN Model 

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

st.cache_data.clear()

# Extract Features and Scale Data
features = ["danceability", "energy", "speechiness", "instrumentalness", "valence"]
X = df_data[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Train KNN Model
knn = NearestNeighbors(n_neighbors=1, metric="euclidean")  
knn.fit(X_scaled)

#Take User Input
#def get_user_input(features):
st.title("Spotify Song Recommendation")
st.write("Enter song characteristics to get recommended tracks!")
danceability = st.slider("Danceability (Low Danceability <-> High Danceability)", 0.0, 1.0, 0.5)
energy = st.slider("Energy (Low Energy <-> High Energy)", 0.0, 1.0, 0.5)
speechiness = st.slider("Speechiness (Low Presence of Words <-> High Presence of Words)", 0.0, 1.0, 0.1)
instrumentalness = st.slider("Instrumentalness (Low Presence of Instruments <-> High Presence of Instruments)", 0.0, 1.0, 0.0)
valence = st.slider("Mood (Sad <-> Happy)", 0.0, 1.0, 0.5)


container = st.container()

if st.button("Find My Song (KNN) :headphones:"):
    user_features = np.array([[danceability, energy, speechiness, instrumentalness, valence]])
    user_features_scaled = scaler.transform(user_features)  

# Find the Best Matching Song
    _, index = knn.kneighbors(user_features_scaled) #enter in slider data here 
    best_match_index = index[0][0]
    best_match = df_raw.iloc[best_match_index][["Track Name", "Artist Name(s)"]]

    #st.success(f"\n:musical_note:<u>Best match for your input</u>  \n:microphone: Track Name: **{best_match[0]}**  \n:female-singer: Artist Name: **{best_match[1]}**")


    container.markdown(f"""
                <div style="background-color:green; padding:10px; border-radius:5px;">
                🎵 <u><b>Best KNN Match For Your Input</b></u> <br> 
                🎤 <b>Track Name:</b> {best_match[0]} <br>
                👩‍🎤 <b>Artist Name:</b> {best_match[1]}
                </div>
                """, unsafe_allow_html=True)

#K-Means Model

# Train K-Means Model
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10) 
df_data["cluster"] = kmeans.fit_predict(X_scaled)
df_raw_copy = df_raw.copy()
df_raw_copy["cluster"] = df_data["cluster"]

#container2 = st.container()
# if st.button("Find My Song (K-Means) :headphones:"):
#     user_features = np.array([[danceability, energy, speechiness, instrumentalness, valence]])
#     user_features_scaled = scaler.transform(user_features)  

# # Find the Best Matching Song
#     cluster_label = kmeans.predict(user_features_scaled)[0] 
#     cluster_songs = df_data[df_data["cluster"]]==cluster_label 
#     best_match = df_raw.iloc[cluster_songs.sample(1)][["Track Name", "Artist Name(s)"]].values[0]

#     #st.success(f"\n:musical_note:<u>Best match for your input</u>  \n:microphone: Track Name: **{best_match[0]}**  \n:female-singer: Artist Name: **{best_match[1]}**")


#     container2.markdown(f"""
#                 <div style="background-color:green; padding:10px; border-radius:5px;">
#                 🎵 <u><b>Best Match For Your Input</b></u> <br> 
#                 🎤 <b>Track Name:</b> {best_match[0]} <br>
#                 👩‍🎤 <b>Artist Name:</b> {best_match[1]}
#                 </div>
#                 """, unsafe_allow_html=True)
    
#     container2 = st.container()

if st.button("Find My Song (K-Means) 🎧"):
    user_features = np.array([[danceability, energy, speechiness, instrumentalness, valence]])
    user_features_scaled = scaler.transform(user_features)

    # Find the Best Matching Song
    cluster_label = kmeans.predict(user_features_scaled)[0]
    cluster_songs = df_raw_copy[df_raw_copy["cluster"] == cluster_label]

    if not cluster_songs.empty:
        best_match = cluster_songs.sample(1)[["Track Name", "Artist Name(s)"]].values[0]

        container.markdown(f"""
            <div style="background-color:green; padding:10px; border-radius:5px; color:white;">
                🎵 <u><b>Best K-means Match For Your Input</b></u> <br> 
                🎤 <b>Track Name:</b> {best_match[0]} <br>
                👩‍🎤 <b>Artist Name:</b> {best_match[1]}
            </div>
        """, unsafe_allow_html=True)
    else:
        container.warning("No match found, try adjusting your input.")

# # Step 3: Streamlit UI
# st.title(":musical_note: K-Means Song Recommendation System")
# # User input sliders
# danceability = st.slider("Danceability (0-1)", min_value=0.0, max_value=1.0, step=0.01, value=0.5)
# energy = st.slider("Energy (0-1)", min_value=0.0, max_value=1.0, step=0.01, value=0.5)
# tempo = st.slider("Tempo (BPM)", min_value=50, max_value=200, step=1, value=120)
# # Function to find closest cluster
# def find_best_cluster(user_input):
#     user_scaled = scaler.transform(user_input)  # Scale input
#     cluster_label = kmeans.predict(user_scaled)[0]  # Find closest cluster
#     return cluster_label
# # Function to recommend a song from the closest cluster
# def recommend_song(cluster_label):
#     cluster_songs = df[df["cluster"] == cluster_label]
#     return cluster_songs.sample(1)["track_name"].values[0]  # Random song from cluster
# # :dart: Step 4: Predict & Recommend
# if st.button("Find My Song :headphones:"):
#     user_features = np.array([[danceability, energy, tempo]])
#     best_cluster = find_best_cluster(user_features)
#     best_match = recommend_song(best_cluster)
#     st.success(f":notes: Best match from Cluster {best_cluster}: **{best_match}** :musical_note:")

# # %%
# #Gaussian Mixture Model

# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.mixture import GaussianMixture
# from sklearn.preprocessing import StandardScaler

# # 🎵 Example Dataset (Replace with actual dataset)
# data = {
#     "track_name": ["Song A", "Song B", "Song C", "Song D", "Song E", "Song F", "Song G"],
#     "danceability": [0.8, 0.6, 0.9, 0.4, 0.7, 0.5, 0.3],
#     "energy": [0.7, 0.5, 0.9, 0.3, 0.6, 0.4, 0.2],
#     "tempo": [120, 130, 110, 100, 125, 140, 90]
# }

# # Convert to DataFrame
# df = pd.DataFrame(data)

# # 🎯 Step 1: Preprocess Features
# features = ["danceability", "energy", "tempo"]
# X = df[features]

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # 🎯 Step 2: Train Gaussian Mixture Model (GMM)
# gmm = GaussianMixture(n_components=3, random_state=42, covariance_type='full')
# df["gmm_cluster"] = gmm.fit_predict(X_scaled)

# # 🎯 Step 3: Streamlit UI
# st.title("🎵 Gaussian Mixture Model Song Recommendation")

# # User input sliders
# danceability = st.slider("Danceability (0-1)", min_value=0.0, max_value=1.0, step=0.01, value=0.5)
# energy = st.slider("Energy (0-1)", min_value=0.0, max_value=1.0, step=0.01, value=0.5)
# tempo = st.slider("Tempo (BPM)", min_value=50, max_value=200, step=1, value=120)

# # Function to find closest cluster
# def find_best_cluster(user_input):
#     user_scaled = scaler.transform(user_input)  # Scale input
#     cluster_label = gmm.predict(user_scaled)[0]  # Predict closest cluster
#     return cluster_label

# # Function to recommend a song from the closest cluster
# def recommend_song(cluster_label):
#     cluster_songs = df[df["gmm_cluster"] == cluster_label]
#     if cluster_songs.empty:
#         return "No match found, try adjusting your input."
#     return cluster_songs.sample(1)["track_name"].values[0]  # Random song from cluster

# # 🎯 Step 4: Predict & Recommend
# if st.button("Find My Song 🎧"):
#     user_features = np.array([[danceability, energy, tempo]])
#     best_cluster = find_best_cluster(user_features)
#     best_match = recommend_song(best_cluster)
    
#     st.success(f"🎶 Best match from GMM: **{best_match}** 🎵")

# # %%
# #DBSAN  

# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.cluster import DBSCAN
# from sklearn.preprocessing import StandardScaler

# # 🎵 Example Dataset (Replace with actual dataset)
# data = {
#     "track_name": ["Song A", "Song B", "Song C", "Song D", "Song E", "Song F", "Song G"],
#     "danceability": [0.8, 0.6, 0.9, 0.4, 0.7, 0.5, 0.3],
#     "energy": [0.7, 0.5, 0.9, 0.3, 0.6, 0.4, 0.2],
#     "tempo": [120, 130, 110, 100, 125, 140, 90]
# }

# # Convert to DataFrame
# df = pd.DataFrame(data)

# # 🎯 Step 1: Preprocess Features
# features = ["danceability", "energy", "tempo"]
# X = df[features]

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # 🎯 Step 2: Train DBSCAN Model
# dbscan = DBSCAN(eps=1.0, min_samples=2)  # Adjust parameters based on data
# df["dbscan_cluster"] = dbscan.fit_predict(X_scaled)

# # 🎯 Step 3: Streamlit UI
# st.title("🎵 DBSCAN Song Recommendation System")

# # User input sliders
# danceability = st.slider("Danceability (0-1)", min_value=0.0, max_value=1.0, step=0.01, value=0.5)
# energy = st.slider("Energy (0-1)", min_value=0.0, max_value=1.0, step=0.01, value=0.5)
# tempo = st.slider("Tempo (BPM)", min_value=50, max_value=200, step=1, value=120)

# # Function to find closest cluster
# def find_best_cluster(user_input):
#     user_scaled = scaler.transform(user_input)  # Scale input
#     cluster_label = dbscan.fit_predict(user_scaled)[0]  # DBSCAN does not predict well
#     return cluster_label

# # Function to recommend a song from the closest cluster
# def recommend_song(cluster_label):
#     cluster_songs = df[df["dbscan_cluster"] == cluster_label]
#     if cluster_songs.empty:
#         return "No match found, try adjusting your input."
#     return cluster_songs.sample(1)["track_name"].values[0]  # Random song from cluster

# # 🎯 Step 4: Predict & Recommend
# if st.button("Find My Song 🎧"):
#     user_features = np.array([[danceability, energy, tempo]])
#     best_cluster = find_best_cluster(user_features)
#     best_match = recommend_song(best_cluster)
    
#     st.success(f"🎶 Best match from DBSCAN: **{best_match}** 🎵")

# # %%
# #Agglomerative Clustering

# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.preprocessing import StandardScaler
# from scipy.spatial.distance import cdist

# # 🎵 Example Dataset (Replace with actual dataset)
# data = {
#     "track_name": ["Song A", "Song B", "Song C", "Song D", "Song E", "Song F", "Song G"],
#     "danceability": [0.8, 0.6, 0.9, 0.4, 0.7, 0.5, 0.3],
#     "energy": [0.7, 0.5, 0.9, 0.3, 0.6, 0.4, 0.2],
#     "tempo": [120, 130, 110, 100, 125, 140, 90]
# }

# # Convert to DataFrame
# df = pd.DataFrame(data)

# # 🎯 Step 1: Preprocess Features
# features = ["danceability", "energy", "tempo"]
# X = df[features]

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # 🎯 Step 2: Train Agglomerative Clustering Model
# agg_clustering = AgglomerativeClustering(n_clusters=3)
# df["agg_cluster"] = agg_clustering.fit_predict(X_scaled)

# # 🎯 Step 3: Streamlit UI
# st.title("🎵 Agglomerative Clustering Song Recommendation")

# # User input sliders
# danceability = st.slider("Danceability (0-1)", min_value=0.0, max_value=1.0, step=0.01, value=0.5)
# energy = st.slider("Energy (0-1)", min_value=0.0, max_value=1.0, step=0.01, value=0.5)
# tempo = st.slider("Tempo (BPM)", min_value=50, max_value=200, step=1, value=120)

# # Function to find closest cluster
# def find_best_cluster(user_input):
#     user_scaled = scaler.transform(user_input)  # Scale input
#     distances = cdist(user_scaled, X_scaled)  # Compute distances to existing points
#     closest_point = np.argmin(distances)  # Find closest song in dataset
#     cluster_label = df.iloc[closest_point]["agg_cluster"]
#     return cluster_label

# # Function to recommend a song from the closest cluster
# def recommend_song(cluster_label):
#     cluster_songs = df[df["agg_cluster"] == cluster_label]
#     if cluster_songs.empty:
#         return "No match found, try adjusting your input."
#     return cluster_songs.sample(1)["track_name"].values[0]  # Random song from cluster

# # 🎯 Step 4: Predict & Recommend
# if st.button("Find My Song 🎧"):
#     user_features = np.array([[danceability, energy, tempo]])
#     best_cluster = find_best_cluster(user_features)
#     best_match = recommend_song(best_cluster)
    
#     st.success(f"🎶 Best match from Agglomerative Clustering: **{best_match}** 🎵")

# # %%


# # %%


# # %%


# # %%
# df_test.head()

# # %%
# # Scale data with Standard Scaler
# scaler = StandardScaler()

# scaled_data = scaler.fit_transform(df_x)

# # call PCA
# pca = PCA(n_components=1)

# # fit and apply
# genres_pca = pca.fit_transform(scaled_data)

# # Create DataFrame with PCA results
# genres_pca_df = pd.DataFrame(
#     genres_pca,
#     columns=['genre_pca_1',
#             #  'genre_pca_2',
#             #  'genre_pca_3',
#             #  'genre_pca_4',
#             #  'genre_pca_5',
#             #  'genre_pca_6',
#             #  'genre_pca_7',
#             #  'genre_pca_8',
#             #  'genre_pca_9',
#             #  'genre_pca_10',
#             #  'genre_pca_11',
#             #  'genre_pca_12',
#             #  'genre_pca_13',
#             #  'genre_pca_14',
#             #  'genre_pca_15',
#             #  'genre_pca_16',
#             #  'genre_pca_17',
#             #  'genre_pca_18',
#             #  'genre_pca_19',
#             #  'genre_pca_20'
#              ])

# genres_pca_df

# # %%
# pca.explained_variance_ratio_

# # %%
# # Sum the explained variance ratios
# total_explained_variance = pca.explained_variance_ratio_.sum()
# # Print the total explained variance
# print(f"Total Explained Variance: {total_explained_variance}")

# # %%


# # %%


# # %%


# # %%


# # %%


# # %%


# # %%
# # # Select only numeric columns for modeling
# # numeric_features = df_cleaned.select_dtypes(include=['float64', 'int64']).columns

# # # Create feature matrix X
# # X = df_cleaned[numeric_features]

# # # Optional: Create new features
# # # Example: Combining features or creating ratios
# # X['energy_valence_ratio'] = X['energy'] / X['valence']

# # %%
# # # Remove the problematic energy_valence_ratio column if it exists
# # if 'energy_valence_ratio' in X.columns:
# #     X = X.drop('energy_valence_ratio', axis=1)

# # # Create the ratio feature with handling for zero values
# # X['energy_valence_ratio'] = X['energy'] / X['valence'].replace(0, 0.0001)  # Replace zeros with small value

# # # Now scale the features
# # scaler = StandardScaler()
# # X_scaled = scaler.fit_transform(X)
# # X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# # %%
# # # PCA for dimensionality reduction
# # pca = PCA(n_components=0.95)  # Keep 95% of variance
# # X_pca = pca.fit_transform(X_scaled)

# # # Or t-SNE for non-linear dimensionality reduction
# # tsne = TSNE(n_components=2, random_state=42)
# # X_tsne = tsne.fit_transform(X_scaled)

# # %%
# # Using IQR method to detect outliers
# def remove_outliers(df, columns):
#     df_clean = df.copy()
#     for col in columns:
#         Q1 = df_clean[col].quantile(0.25)
#         Q3 = df_clean[col].quantile(0.75)
#         IQR = Q3 - Q1
#         lower_bound = Q1 - 1.5 * IQR
#         upper_bound = Q3 + 1.5 * IQR
#         df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
#     return df_clean

# # Apply outlier removal
# X_no_outliers = remove_outliers(X_scaled, X_scaled.columns)

# # %%
# #unique_genres = set(df_cleaned['Artist Genres'].str.split(',').explode().value_counts())
# unique_genres = df_cleaned['artist__genres'].str.split(',').explode().value_counts()
# print(len(unique_genres))
# print(unique_genres.head(20))

# # %%
# # Looking at the error message and available columns, we see that 'Album Genres' doesn't exist
# # Let's use 'artist__genres' instead since we already have that data

# # Get genre counts from the already exploded artist__genres
# genre_counts = df_cleaned['artist__genres'].value_counts()

# # Select top N genres (e.g., top 20)
# top_n_genres = 20
# top_genres = genre_counts.head(top_n_genres).index

# # Create dummies only for top genres
# genre_dummies = pd.get_dummies(
#     df_cleaned['artist__genres'].where(df_cleaned['artist__genres'].isin(top_genres), 'other'),
#     prefix='genre'
# )

# # Group by index and join with original dataframe
# genre_dummies = genre_dummies.groupby(df_cleaned.index).sum()
# df_with_top_genres = pd.concat([df_cleaned, genre_dummies], axis=1)

# # No need to drop 'Album Genres' since it doesn't exist
# # df_with_top_genres = df_with_top_genres.drop('Album Genres', axis=1)

# print("\nShape with top genres only:", df_with_top_genres.shape)
# print("\nTop genre columns:", genre_dummies.columns.tolist())

# # %%
# df_with_top_genres.info()

# # %%
# df_with_top_genres.head()

# # %%
# # Test the models
# # Unsupervised models K-means, Gaussian 

# # %%
# # visualize model accuracy
# # the elbow thing
# # mushroom pizza
# # 

# # %%


# # %%


# # %%


# # %%



