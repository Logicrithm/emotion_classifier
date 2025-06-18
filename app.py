import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

# Load tokenizer and label encoder
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

with open("label_encoder.pkl", "rb") as enc:
    label_encoder = pickle.load(enc)

# Load the trained model
model = tf.keras.models.load_model("emotion_model.h5")

# Max length used during training
MAX_LEN = 100

# App Title
st.title("🧠 Emotion Classifier")
st.write("Enter a sentence and the model will predict the underlying emotion.")

# User input
user_input = st.text_area("Enter text:", "")

# Predict button
if st.button("Predict Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocess input
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=MAX_LEN)

        # Predict
        prediction = model.predict(padded)
        predicted_class = np.argmax(prediction)
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]

        # Show results
        st.success(f"🗣 Emotion: **{predicted_label.capitalize()}**")
        st.bar_chart(prediction[0])
