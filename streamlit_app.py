import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model, scaler, and pca from the single pickle file
try:
    loaded_objects = joblib.load('model_scaler_pca2.pkl')
    model = loaded_objects['model']
    scaler = loaded_objects['scaler']
    pca = loaded_objects['pca']
    feature_names = loaded_objects['feature_names']
    st.success("Model, scaler, and PCA loaded successfully!")
except FileNotFoundError:
    st.error("Error: 'model_scaler_pca2.pkl' not found. Please make sure the file is in the same directory.")
    st.stop()
except KeyError:
    st.error("Error: 'model_scaler_pca2.pkl' does not contain expected objects. Please check the file content.")
    st.stop()


st.title('Song Popularity Predictor')

st.write("""
This app predicts the popularity of a song based on its audio features.
""")

# Input features from the user
st.sidebar.header('Song Features')

def user_input_features():
    acousticness = st.sidebar.slider('Acousticness', 0.0, 1.0, 0.5)
    danceability = st.sidebar.slider('Danceability', 0.0, 1.0, 0.5)
    energy = st.sidebar.slider('Energy', 0.0, 1.0, 0.5)
    instrumentalness = st.sidebar.slider('Instrumentalness', 0.0, 1.0, 0.5)
    key = st.sidebar.slider('Key', 0, 11, 5)
    liveness = st.sidebar.slider('Liveness', 0.0, 1.0, 0.5)
    loudness = st.sidebar.slider('Loudness', -60.0, 0.0, -30.0)
    audio_mode = st.sidebar.radio('Audio Mode', [0, 1])
    speechiness = st.sidebar.slider('Speechiness', 0.0, 1.0, 0.5)
    tempo = st.sidebar.slider('Tempo', 0.0, 250.0, 120.0)
    time_signature = st.sidebar.slider('Time Signature', 0, 5, 4)
    audio_valence = st.sidebar.slider('Audio Valence', 0.0, 1.0, 0.5)
    song_duration_ms = st.sidebar.slider('Song Duration (ms)', 0, 1000000, 200000)


    data = {'song_duration_ms': song_duration_ms,
            'acousticness': acousticness,
            'danceability': danceability,
            'energy': energy,
            'instrumentalness': instrumentalness,
            'key': key,
            'liveness': liveness,
            'loudness': loudness,
            'audio_mode': audio_mode,
            'speechiness': speechiness,
            'tempo': tempo,
            'time_signature': time_signature,
            'audio_valence': audio_valence}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Ensure the order of columns matches the training data and convert to numpy array
# Explicitly reindex the input_df to match the order of feature_names
input_data = input_df.reindex(columns=feature_names).values


# Display the input features
st.subheader('User Input Features')
st.write(input_df)

# Make prediction
if st.sidebar.button('Predict Popularity'):
    try:
        # Scale the input features
        scaled_input = scaler.transform(input_data)

        # Apply PCA transformation
        pca_input = pca.transform(scaled_input)

        prediction = model.predict(pca_input)
        st.subheader('Predicted Song Popularity')
        # Assuming popularity is on a scale of 0-100
        st.write(f"{prediction[0]:.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
