# Emotion Classifier 🤖

This is an LSTM-based emotion classification model trained on a labeled dataset of sentences to predict emotions like **joy, sadness, anger, fear, love, surprise**.

### 🔍 Dataset
- Source: [Kaggle Emotion Dataset](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)
- Preprocessed into training and validation sets

### 🧠 Model
- **Embedding layer** + **LSTM**
- **Dense** + **Softmax** for classification
- Achieved ~90% validation accuracy

### 🚀 Deployment
- Deployed using **Streamlit Cloud**
- [🖥 View Live App](https://emotionclassifier-tjayjmiugzk9qnhv7fqb3z.streamlit.app/)

### 📦 Requirements
```bash
streamlit
tensorflow
numpy
