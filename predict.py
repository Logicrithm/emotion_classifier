import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = load_model('emotion_model.h5')

# Load tokenizer and encoder
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Define predict function
def predict_emotion(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=50, padding='post', truncating='post')
    pred = model.predict(padded)
    emotion = label_encoder.inverse_transform([np.argmax(pred)])
    return emotion[0]

# Try it
while True:
    user_input = input("Type a sentence (or 'exit'): ")
    if user_input.lower() == "exit":
        break
    emotion = predict_emotion(user_input)
    print("💡 Emotion:", emotion)
