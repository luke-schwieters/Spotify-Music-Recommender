import argparse
import logging
import numpy as np
import sklearn
from sklearn.decomposition import PCA
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
from collections import Counter
import streamlit as st
import os

logger = logging.getLogger('clustering')
logging.basicConfig(level=logging.INFO)

# Spotify Credintals
#SPOTIPY_CLIENT_ID = ""
#SPOTIPY_CLIENT_SECRET = ""
SPOTIPY_REDIRECT_URI = "http://localhost/"

SPOTIPY_CLIENT_ID = st.text_input('Spotify Client ID', '', type='password')
SPOTIPY_CLIENT_SECRET = st.text_input('Spotify Client Secret', '', type='password')

# Create Spotifiy Session wth Oauth, using Streamlit Cache Could help with login Loops
# @st.cache_data
def create_client():
    #print("Authentication")
    sp_oauth = SpotifyOAuth(
        client_id=SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET,
        redirect_uri=SPOTIPY_REDIRECT_URI,
        scope="user-modify-playback-state playlist-modify-public",
    )

    # Return Spotify client
    return spotipy.Spotify(auth_manager=sp_oauth)
sp = create_client()

#Used for Debugging, Sets Defaults Values
def get_args():
    parser = argparse.ArgumentParser(description='Cluster songs from a Spotify playlist')
    parser.add_argument('-p', '--playlist', default="0nEeE93iD7X6d5dh271wCT", help='Spotify Playlist ID')
    parser.add_argument('-k', '--clusters', type=int, default=5, help='Number of clusters for KMeans Clustering')
    return parser.parse_args()

#Returns list of tracks in given playlist ID, that is not a local file
def get_playlist_tracks(playlist_id):
    #Using "next" value to call larger playists
    tracks = []
    result = sp.playlist_items(playlist_id, additional_types=['track'])
    tracks.extend(result['items'])
    while result['next']:
        result = sp.next(result)
        tracks.extend(result['items'])
    return [item for item in tracks if not item['is_local']]

def fetch_audio_features(track_uris):
    #Must get artist in chunks of 100 from Spotify
    chunks = [track_uris[i:i + 100] for i in range(0,len(track_uris),100)]
    all_features = []
    for chunk in chunks:
        response = sp.audio_features(chunk)
        all_features.extend(response)
    #print(response)
    return all_features

def calculate_average_features(features):
    stats = {
        'danceability': [],
        'energy': [],
        'key': [],
        'loudness': [],
        'speechiness': [],
        'acousticness': [],
        'instrumentalness': [],
        'liveness': [],
        'valence': [],
        'tempo': [],
        'duration_ms': [],
        'time_signature': [],
        'mode': []
    }
    for feature in features:
        for key in stats.keys():
            stats[key].append(feature[key])
    return {key: np.mean(values) for key, values in stats.items()}

def get_all_artists(tracks):
    #Must get artist in chunks of 50 from Spotify
    artists_uri = [item['track']['artists'][0]['uri'] for item in tracks]
    chunks = [artists_uri[i:i + 50] for i in range(0, len(artists_uri), 50)]
    
    all_artists = []
    for chunk in chunks:
        artists_response = sp.artists(chunk)
        all_artists.extend(artists_response['artists'])
    return all_artists

#Calculates the Euclidean distance between feature vectors, from average vector
def calculate_distance(feature, average_features):
    return np.sqrt(sum((feature[key] - average_features[key]) ** 2 for key in average_features))

def select_representative_tracks(features, track_uris, num_seeds=5):
    average_features = calculate_average_features(features)
    distances = []
    for i in range(len(track_uris)):
        track_uri = track_uris[i]
        feature = features[i]
        distance = calculate_distance(feature, average_features)
        distances.append((track_uri, distance))
    
    sorted_distances = sorted(distances, key=lambda x: x[1])
    # Select the top num_seeds tracks
    representative_tracks = []
    for i in range(min(num_seeds, len(sorted_distances))):
        representative_tracks.append(sorted_distances[i][0])
    
    return representative_tracks
#Getting Recommendation based on Average Features,and Seed Tracks
def get_recommended_songs(average_features, seed_tracks,popularity):
    recommendations = sp.recommendations(
        max_popularity=popularity,
        limit=12,
        seed_tracks=seed_tracks,
        target_danceability=average_features['danceability'],
        target_energy=average_features['energy'],
        target_loudness=average_features['loudness'],
        target_speechiness=average_features['speechiness'],
        target_acousticness=average_features['acousticness'],
        target_instrumentalness=average_features['instrumentalness'],
        target_liveness=average_features['liveness'],
        target_valence=average_features['valence'],
        target_tempo=average_features['tempo']
    )
    return recommendations['tracks']
#Getting Recommendation based onSeed Tracks
def get_recommended_songs_tracks(seed_tracks,popularity):
    recommendations = sp.recommendations(seed_tracks=seed_tracks, max_popularity=popularity, limit=12)
    return recommendations['tracks']
#Getting Recommendations based on seed Geners and Average features
def get_recommended_songs_genres(average_features, seed_genres,popularity):
    recommendations = sp.recommendations(
        max_popularity=popularity,
        limit=12,
        seed_genres=seed_genres,
        target_danceability=average_features['danceability'],
        target_energy=average_features['energy'],
        target_loudness=average_features['loudness'],
        target_speechiness=average_features['speechiness'],
        target_acousticness=average_features['acousticness'],
        target_instrumentalness=average_features['instrumentalness'],
        target_liveness=average_features['liveness'],
        target_valence=average_features['valence'],
        target_tempo=average_features['tempo']
    )
    return recommendations['tracks']

#Cluster using a spectral method
def cluster_songs(features, num_clusters,n_components = 8):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(scaled_features)
    spectral = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', assign_labels='kmeans', random_state=42)
    labels = spectral.fit_predict(pca_features)
    return labels

def add_to_queue(uri):
    try:
        sp.add_to_queue(uri)
        st.toast(f"Added to queue.", icon="✅")
    except Exception as e:
        st.toast(f"Error adding {uri} to queue: {str(e)}", icon="❗")

def add_to_playlist(uri, playlist_id):
    try:
        sp.playlist_add_items(playlist_id, [uri])
        st.toast(f"Added to playlist.", icon="✅")
    except Exception as e:
        st.toast(f"Error adding {uri} to playlist: {str(e)}", icon="❗")

# initialize session state for average features and recommendations
if 'average_features' not in st.session_state:
    st.session_state['average_features'] = None

if 'final_recommendations' not in st.session_state:
    st.session_state['final_recommendations'] = []

#Initialize session state for slider values
if 'slider_values' not in st.session_state:
    st.session_state['slider_values'] = {
        'danceability': 0.5,
        'energy': 0.5,
        'loudness': -30.0,
        'speechiness': 0.5,
        'acousticness': 0.5,
        'instrumentalness': 0.5,
        'liveness': 0.5,
        'valence': 0.5,
        'tempo': 125.0
    }

st.title("Spotify New Music Recommeneder")
playlist_id = st.text_input('Spotify Playlist Link', '0nEeE93iD7X6d5dh271wCT',help="Get your playlist link by sharing playlist and copying link. Using Playlist URI, or ID also works")
num_clusters = st.slider(label="Clusters (Less Clusters = Higher Song Tolerance)", min_value=2, max_value=35, value=10,help="This defines the number of clustered groupings your playlist will be split into, The more Clusters less output songs")
popularity = st.slider(label="Popularity of Generated Songs", min_value=0, max_value=100, value=100,help="Generally, Values under 40, will give songs considered pretty Niche, values lowers may return results with no songs")


#Function Used to Calculate an ideal number toleated output clusters form given number.
number_of_output_clusters = int(1.54*pow((num_clusters),0.309))
#print(number_of_output_clusters)

#Add sliders to sidebar with default ranges
def slider_update():
    with st.sidebar:
        st.header("Feature Ranges")
        st.session_state['slider_values']['danceability'] = st.slider(
            'Danceability', 0.0, 1.0, st.session_state['slider_values']['danceability'])
        st.session_state['slider_values']['energy'] = st.slider(
            'Energy', 0.0, 1.0, st.session_state['slider_values']['energy'])
        st.session_state['slider_values']['loudness'] = st.slider(
            'Loudness', -60.0, 0.0, st.session_state['slider_values']['loudness'])
        st.session_state['slider_values']['speechiness'] = st.slider(
            'Speechiness', 0.0, 1.0, st.session_state['slider_values']['speechiness'])
        st.session_state['slider_values']['acousticness'] = st.slider(
            'Acousticness', 0.0, 1.0, st.session_state['slider_values']['acousticness'])
        st.session_state['slider_values']['instrumentalness'] = st.slider(
            'Instrumentalness', 0.0, 1.0, st.session_state['slider_values']['instrumentalness'])
        st.session_state['slider_values']['liveness'] = st.slider(
            'Liveness', 0.0, 1.0, st.session_state['slider_values']['liveness'])
        st.session_state['slider_values']['valence'] = st.slider(
            'Valence', 0.0, 1.0, st.session_state['slider_values']['valence'])
        st.session_state['slider_values']['tempo'] = st.slider(
            'Tempo', 0.0, 250.0, st.session_state['slider_values']['tempo'])
        
slider_update()

if st.button('Generate Recommendations', type='primary'):
    #Fetch playlist tracks
    tracks = get_playlist_tracks(playlist_id)
    track_uris = [item['track']['uri'] for item in tracks]

    #Fetch audio features
    #print(len(track_uris))
    features = fetch_audio_features(track_uris)

    # Select 5 representative tracks based on Echulidan distance from average feature vector
    representative_tracks = select_representative_tracks(features, track_uris)

    #Calculate average features for side bar
    average_features = calculate_average_features(features)
    st.session_state['average_features'] = average_features

    # Update slider values with the calculated averages
    st.session_state['slider_values']['danceability'] = average_features['danceability']
    st.session_state['slider_values']['energy'] = average_features['energy']
    st.session_state['slider_values']['loudness'] = average_features['loudness']
    st.session_state['slider_values']['speechiness'] = average_features['speechiness']
    st.session_state['slider_values']['acousticness'] = average_features['acousticness']
    st.session_state['slider_values']['instrumentalness'] = average_features['instrumentalness']
    st.session_state['slider_values']['liveness'] = average_features['liveness']
    st.session_state['slider_values']['valence'] = average_features['valence']
    st.session_state['slider_values']['tempo'] = average_features['tempo']

    #Getting all genres of playlist from each artist, and finding the 5 most common to use as seed for reccomendations
    all_artists = get_all_artists(tracks)
    all_genres = [artist['genres'] for artist in all_artists]
    flat_genres = [genre for sublist in all_genres for genre in sublist]
    genre_counts = Counter(flat_genres)
    top_5_genres = [genres for genres, _ in genre_counts.most_common(5)]

    #Get recommended songs based on average features and representative tracks, and seed geners
    recommended_songs = (
        get_recommended_songs(average_features, representative_tracks,popularity)
        + get_recommended_songs_tracks(representative_tracks,popularity)
        + get_recommended_songs_genres(average_features, top_5_genres,popularity)
    )

    #Removing Duplicates songs
    unique_songs = []
    seen_uris = set()
    for song in recommended_songs:
        uri = song['uri']
        if uri not in seen_uris:
            seen_uris.add(uri)
            unique_songs.append(song)

    recommended_songs = unique_songs
    recommended_uris = [track['uri'] for track in recommended_songs]
    
    #Get audio features for recommended songs
    recommended_features = fetch_audio_features(recommended_uris)

    # feature matrix for clustering
    all_features = features + recommended_features
    feature_matrix = np.array([
        [
            feature['danceability'],
            feature['energy'],
            feature['loudness'],
            feature['speechiness'],
            feature['acousticness'],
            feature['instrumentalness'],
            feature['liveness'],
            feature['valence'],
            feature['tempo'],
        ]
        for feature in all_features if feature is not None
    ])

    # Cluster songs
    labels = cluster_songs(feature_matrix, num_clusters)
    #print(labels)

    # Separate playlist and recommended song labels
    print(len(labels))
    print(len(tracks))
    playlist_labels = labels[:len(tracks)]
    recommended_labels = labels[len(tracks):]
    
    #print(recommended_labels)

    #Count the number of playlist songs in each cluster
    playlist_cluster_counts = Counter(playlist_labels)

    # Find the top 3 most common clusters for playlist songs
    most_common_clusters = [cluster for cluster, _ in playlist_cluster_counts.most_common(3)]

    #grab the recommended songs in the top 3 most common clusters
    recommended_in_common_cluster = [
        recommended_songs[i]
        for i in range(len(recommended_songs))
        if recommended_labels[i] in most_common_clusters
    ]

    st.session_state['final_recommendations'] = recommended_in_common_cluster[:20]

    #Log final recommendations
    for track in st.session_state['final_recommendations']:
        logger.info('Recommended Track: %s - %s', track['name'], track['artists'][0]['name'])

    st.rerun()

#Seperate button to get songs using custom feature values
if st.button('Generate with Custom Values'):
    #Get custom values from sliders
    custom_features = {key: value for key, value in st.session_state['slider_values'].items()}

    tracks = get_playlist_tracks(playlist_id)
    track_uris = [item['track']['uri'] for item in tracks]

    features = fetch_audio_features(track_uris)

    representative_tracks = select_representative_tracks(features, track_uris)

    all_artists = get_all_artists(tracks)
    all_genres = [artist['genres'] for artist in all_artists]
    flat_genres = [genre for sublist in all_genres for genre in sublist]
    genre_counts = Counter(flat_genres)
    top_5_genres = [genres for genres, _ in genre_counts.most_common(5)]

    recommended_songs = (
        get_recommended_songs(custom_features, representative_tracks,popularity)
        + get_recommended_songs_tracks(representative_tracks,popularity)
        + get_recommended_songs_genres(custom_features, top_5_genres,popularity)
    )
    unique_songs = []
    seen_uris = set()
    for song in recommended_songs:
        uri = song['uri']
        if uri not in seen_uris:
            seen_uris.add(uri)
            unique_songs.append(song)
    recommended_songs = unique_songs
    
    recommended_uris = [track['uri'] for track in recommended_songs]

    recommended_features = fetch_audio_features(recommended_uris)

    all_features = features + recommended_features
    feature_matrix = np.array([
        [
            feature['danceability'],
            feature['energy'],
            feature['loudness'],
            feature['speechiness'],
            feature['acousticness'],
            feature['instrumentalness'],
            feature['liveness'],
            feature['valence'],
            feature['tempo'],
        ]
        for feature in all_features if feature is not None
    ])

    labels = cluster_songs(feature_matrix, num_clusters)

    playlist_labels = labels[:len(tracks)]
    recommended_labels = labels[len(tracks):]
    playlist_cluster_counts = Counter(playlist_labels)

    most_common_clusters = [cluster for cluster, _ in playlist_cluster_counts.most_common(number_of_output_clusters)]

    recommended_in_common_cluster = [
        recommended_songs[i]
        for i in range(len(recommended_songs))
        if recommended_labels[i] in most_common_clusters
    ]

    st.session_state['final_recommendations'] = recommended_in_common_cluster[:20]

    for track in st.session_state['final_recommendations']:
        logger.info('Recommended Track: %s - %s', track['name'], track['artists'][0]['name'])

#Display final recommendations with add to queue and add to playlist buttons
if 'final_recommendations' in st.session_state and st.session_state['final_recommendations']:
    st.write("Recommended Tracks:")
    col1, col2, col3 = st.columns([6, 2, 2], gap="small")

    #maximum length for the button text
    max_length_track = 35 
    max_length_artist= 20

    #buttons for recommended tracks
    for track in st.session_state['final_recommendations']:
        #Truncate the song title and artist name if they are too long
        track_name=f"{track['name']}"
        track_artist = f"{track['artists'][0]['name']}"

        if len(track_name) > max_length_track:
            track_name = track_name[:max_length_track - 3] + "..."
        if len(track_artist) > max_length_artist:
            track_artist = track_artist[:max_length_artist - 3] + "..."
        truncated_text = track_name + " by "+ track_artist

        with col1:
            if st.button(truncated_text, key=f"songlink-{track['uri']}",help=f"{track['name']} by {track['artists'][0]['name']}", use_container_width=True):
                add_to_queue(track['uri'])
        with col2:
            if st.button("Add to Queue", key=f"queue-{track['uri']}",help=f"Add to Queue - {track['name']} by {track['artists'][0]['name']}"):
                add_to_queue(track['uri'])
        with col3:
            if st.button("Add to Playlist", key=f"playlist-{track['uri']}",help=f"Add to Playlist - {track['name']} by {track['artists'][0]['name']}"):
                add_to_playlist(track['uri'], playlist_id)
