import streamlit as st
from utils import extract_features, load_tflite_model, predict
import tempfile
import os

# Define class names (update if using different genres)
class_names = ['blues', 'classical', 'country', 'disco', 'hiphop',
               'jazz', 'metal', 'pop', 'reggae', 'rock']

st.title("🎵 Music Genre Classifier (TFLite)")
st.write("Upload a `.wav` file and I'll try to guess the genre!")

# Load TFLite model
model = load_tflite_model('model_reduced.tflite')

uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        features = extract_features(tmp_path)
        predicted_index, probabilities = predict(model, features)
        predicted_genre = class_names[predicted_index]
        st.success(f"🎶 Predicted Genre: **{predicted_genre}**")
    except Exception as e:
        st.error(f"Something went wrong: {e}")
    finally:
        os.remove(tmp_path)