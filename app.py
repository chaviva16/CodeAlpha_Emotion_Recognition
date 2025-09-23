import streamlit as st
import numpy as np
import librosa
import pickle
from tensorflow.keras.models import load_model

# =========================
# 🎨 Page Config
# =========================
st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎤", layout="centered")

# =========================
# 📌 Load Model and Label Encoder
# =========================
@st.cache_resource
def load_model_and_encoder():
    model = load_model("best_emotion_model.keras")  # new improved model
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le

model, le = load_model_and_encoder()


# =========================
# 📌 MFCC Feature Extraction
# =========================
def extract_features(file_path, max_pad_len=174, n_mfcc=40):
    # Match training: use kaiser_fast resampling
    audio, sample_rate = librosa.load(file_path, res_type="kaiser_fast")
    
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)

    # Pad or truncate to fixed length
    if mfccs.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0,0),(0,pad_width)), mode="constant")
    else:
        mfccs = mfccs[:, :max_pad_len]

    return mfccs[..., np.newaxis]  # shape (40,174,1)

# =========================
# 📌 Prediction Function
# =========================
def predict_emotion(file_path):
    features = extract_features(file_path)
    features = np.expand_dims(features, axis=0)
    prediction = model.predict(features, verbose=0)
    predicted_idx = np.argmax(prediction, axis=1)[0]
    predicted_label = le.inverse_transform([predicted_idx])[0]
    return predicted_label, prediction[0]

# =========================
# 🎤 Streamlit UI
# =========================
st.title("🎤 Speech Emotion Recognition App")
st.write("Upload a short `.wav` file (3–4 seconds) and let the model detect the emotion.")

uploaded_file = st.file_uploader("📂 Choose an audio file", type=["wav"])

if uploaded_file is not None:
    temp_path = "temp.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    
    st.audio(temp_path, format="audio/wav")
    
    with st.spinner("🔍 Analyzing audio..."):
        emotion, probabilities = predict_emotion(temp_path)

    # Emojis for emotions
    emotion_icons = {
        "neutral": "😐",
        "calm": "😌",
        "happy": "😄",
        "sad": "😢",
        "angry": "😡",
        "fearful": "😨",
        "disgust": "🤢",
        "surprised": "😲"
    }
    icon = emotion_icons.get(emotion, "❓")
    
    # Show results
    st.success(f"### Predicted Emotion: {icon} {emotion.capitalize()}")
    
    # Show confidence
    st.subheader("Prediction Confidence")
    prob_dict = dict(zip(le.classes_, probabilities))
    st.bar_chart(prob_dict)
