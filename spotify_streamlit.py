import config
import spotipy as sp
import json
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import pickle
from IPython.display import display, IFrame, clear_output
import streamlit as st

# Set page configuration
import streamlit as st

# Set page configuration
st.set_page_config(page_title='IronHack Song Recommender', page_icon='spotify', layout='wide', initial_sidebar_state='collapsed')

# Define page layout
col1, col2 = st.columns([1, 4])
with col1:
    st.image('spotify-logo.png', width=100)
with col2:
    st.title('Welcome to the IronHack Song Recommender')
    st.markdown('---')

sp = sp.Spotify(auth_manager=SpotifyClientCredentials(client_id= config.client_id,
                                                           client_secret= config.client_secret))

#uploading the dataset with our cluster column
audio_features = pd.read_csv("audio_features.csv")

#function to load the scaler and K-means pickles
def load(filename = "filename.pickle"): 
    try: 
        with open(filename, "rb") as f: 
            return pickle.load(f) 
        
    except FileNotFoundError: 
        print("File not found!")


# function to get user input and search for songs
def search_song():
    song_input = st.text_input("Please enter the song with the name of artist/group that you like (please enter the title and the artist/group): ")
    if song_input:
        results = sp.search(q=song_input, limit=20)
        return results
    return None

# function to display songs and get user feedback

# function to display songs and get user feedback
def display_song(results):
    for i in range(0, 19):
        track_id = results['tracks']['items'][i]['id']
        st.write("Please listen to the song below and answer the following question:")
        st.write(f"Is this the song you're looking for?")

        st.write(IFrame(src="https://open.spotify.com/embed/track/"+track_id+"?utm_source=generator",
                   width="320",
                   height="80",
                   frameborder="0",
                   allowtransparency="true",
                   allow="encrypted-media",
                  ))

        good_song = st.selectbox("", options=["Yes", "No"], key=f"{i}")

        while good_song == "":
            good_song = st.selectbox("Please select an option", options=["Yes", "No"], key="unique_key")
        good_song = good_song.lower()

        if good_song == "yes":
            return track_id

    st.write("Sorry, we couldn't find the song you were looking for.")
    return None

# function to extract song features, scale them, and predict cluster
def predict_cluster(track_id):
    select_song_features = sp.audio_features(track_id)
    del select_song_features[0]['key']
    del select_song_features[0]['mode']
    del select_song_features[0]['type']
    del select_song_features[0]['uri']
    del select_song_features[0]['track_href']
    del select_song_features[0]['analysis_url']
    del select_song_features[0]['time_signature']
    
    # convert to dataframe
    song_df = pd.DataFrame(select_song_features, index=[0])
    
    # remove id for clustering
    song_df.drop(columns="id", inplace=True)
    
    # scale features
    scaler_new_song = load("Model/scaler_song.pickle")
    scaled_song_df = pd.DataFrame(scaler_new_song.transform(song_df), columns=song_df.columns)
    
    # predict cluster
    kmeans_new_song = load("Model/kmeans_10.pickle")
    cluster = kmeans_new_song.predict(scaled_song_df)[0]
    return cluster

# function to recommend a song from the predicted cluster
def recommend_song(cluster):
    #audio_features = pd.read_csv(audio_features.csv")
    random_from_cluster = audio_features[audio_features['cluster'] == cluster].sample()
    id_recommended = random_from_cluster['id'].values[0]
    return id_recommended

# function to display the recommended song
def display_recommended_song(id_recommended):
    st.write("Recommended Song:")
    st.write(IFrame(src="https://open.spotify.com/embed/track/"+id_recommended+"?utm_source=generator",
           width="320",
           height="80",
           frameborder="0",
           allowtransparency="false",
           allow="encrypted-media",
          ))

# main function to run the entire program
def main():
    results = search_song()
    if results is not None:
        track_id = display_song(results)
        if track_id is not None:
            cluster = predict_cluster(track_id)
            id_recommended = recommend_song(cluster)
            display_recommended_song(id_recommended)
            
            like_song = st.selectbox("Do you like this song?", options=["Yes", "No"])
            if like_song == "No":
                id_recommended = recommend_song(cluster)
                display_recommended_song(id_recommended)

# run the program
main()