import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and assets
model = load_model("emotion_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# App UI
st.title("🧠 Emotion Classifier")
st.write("Enter a sentence to predict the emotion it carries.")

user_input = st.text_area("Your sentence here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=50, padding='post', truncating='post')
        prediction = model.predict(padded)
        predicted_emotion = label_encoder.inverse_transform([np.argmax(prediction)])
        
        st.success(f"💬 Predicted Emotion: **{predicted_emotion[0]}**")
        st.bar_chart(prediction[0])
