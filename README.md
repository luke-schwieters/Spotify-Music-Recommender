# Spotify New Music Recommender

## Overview
Welcome to the Spotify New Music Recommender! This project leverages Spotify's API to analyze your playlists and recommend new songs that match your musical preferences. It provides an interactive interface using Streamlit, allowing you to adjust various parameters to fine to the reccommendation it provides you.

## Features
- **Spotify Integration**: Seamlessly integrates with Spotify using OAuth to access your playlists and fetch song features.
- **Clustering**: Uses machine learning clustering techniques to group similar songs.
- **Customization**: Allows users to adjust audio feature ranges to fine-tune recommendations.
- **Recommendations**: Generates recommendations based on audio features, seed tracks, and genres.
- **Listening**: Built in add to queue and add to playlist buttons for easy listening.

## Objective
- **Bring Back Playlist Radios**: Inspired by the removed Spotify playlist radios, this app provides a modern alternative to help you find new music based on your favorite playlists.
- **Discover New Music**: This app aims to help you discover new music that matches your taste by analyzing your existing playlists.

## How It Works
1. **Spotify Authentication**: The app authenticates with Spotify using OAuth. Users must create a Spotify Developer account and enter their credentials.
2. **Playlist Analysis**: The selected playlist is analyzed to extract audio features for each track.
3. **Clustering**: Songs are clustered based on their audio features using Spectral Clustering via kmeans and PCA.
4. **Feature Averages**: Calculates average audio features for the playlist and displays them using sliders for customization.
5. **Recommendations**: Generates song recommendations based on:
    - Average features and representative tracks.
    - Seed tracks that are closest in distance to average feature vector.
    - Most common genres in the playlist.
6. **Display and Interaction**: Displays recommended songs with options to add them to the queue or a playlist.

## Setup and Installation

### Prerequisites
- **Spotify Developer Account**: Create a Spotify Developer account and set up a new application to get your Client ID and Client Secret.
- **Libraries**: Streamlit, SciKit-Learn and Numpy

### Installation
1. **Clone the Repository or Download Script**
2. **Install Dependencies**
3. **Spotify Credentials**: Fill in CLIENT_ID, CLIENT_SECRET and REDIRECT_URI with your info4
4. **Run using streamlit run app.py**

## Using the app

1. **Authentication**: Log in to your Spotify account when prompted. You may need to past redirect link into console
2. **Playlist Input**: Enter your Spotify playlist link, URI, or ID in the provided input box.
3. **Adjust Clusters and Popularity**: Use the sliders to set the number of clusters and the popularity of generated songs. More info given in tooltips.
4. **Feature Customization**: Adjust the audio feature ranges using the sliders in the sidebar.
5. **Generate Recommendations**: Click the "Generate Recommendations" button to get song recommendations based on your playlist.
6. **Add Songs**: Use the buttons to add recommended songs to your queue or playlist.

## Images
![image](https://github.com/Ukelay11/Spotify-Music-Recommender/assets/82103885/b5bb014c-d6c3-4226-9077-466dd739c81a)
![image](https://github.com/Ukelay11/Spotify-Music-Recommender/assets/82103885/85149191-5b12-42da-8e7a-dcd654b519e9)

