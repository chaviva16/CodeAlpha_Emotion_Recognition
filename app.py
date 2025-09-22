import streamlit as st
import numpy as np
import librosa
import pickle
from tensorflow.keras.models import load_model
import threading
import tempfile
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import soundfile as sf

# =========================
# 🎨 Page Config
# =========================
st.set_page_config(page_title="VoiceMood 🎤", page_icon="🎵", layout="centered")

st.title("🎤 VoiceMood: Real-Time Speech Emotion Recognition")
st.write("Upload a `.wav` file or record your voice directly for instant emotion prediction!")

# =========================
# 📌 Load Model and Label Encoder
# =========================
@st.cache_resource
def load_model_and_encoder():
    model = load_model("best_cnn_emotion_model.keras")
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le

model, le = load_model_and_encoder()

# =========================
# 📌 Feature Extraction
# =========================
def extract_features(file_path, max_pad_len=174, n_mfcc=40):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    if mfccs.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0,0),(0,pad_width)), mode="constant")
    else:
        mfccs = mfccs[:, :max_pad_len]
    return mfccs[..., np.newaxis]

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
# 📂 Upload Audio
# =========================
uploaded_file = st.file_uploader("📂 Choose an audio file", type=["wav"])
if uploaded_file is not None:
    temp_path = "temp.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(temp_path, format="audio/wav")

    result = {}
    def run_prediction():
        emotion, probabilities = predict_emotion(temp_path)
        result['emotion'] = emotion
        result['probabilities'] = probabilities

    thread = threading.Thread(target=run_prediction)
    thread.start()
    with st.spinner("🔍 Analyzing audio..."):
        thread.join()

    # Display results
    emotion = result['emotion']
    probabilities = result['probabilities']

    emotion_icons = {
        "neutral": "😐", "calm": "😌", "happy": "😄",
        "sad": "😢", "angry": "😡", "fearful": "😨",
        "disgust": "🤢", "surprised": "😲"
    }
    icon = emotion_icons.get(emotion, "❓")
    st.success(f"### Predicted Emotion: {icon} {emotion.capitalize()}")
    st.subheader("Prediction Confidence")
    st.bar_chart(dict(zip(le.classes_, probabilities)))

# =========================
# 🎙️ Live Recording
# =========================
st.write("🎙️ Or record live audio:")

class LiveAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []
        self.predicted = False
        self.silence_counter = 0
        self.silence_threshold = 20  # frames to detect pause
        self.amplitude_threshold = 1000  # silence amplitude threshold

    def recv_audio(self, frame):
        audio_np = frame.to_ndarray()
        self.frames.append(audio_np)

        # Calculate mean amplitude
        mean_amp = np.mean(np.abs(audio_np))
        if mean_amp < self.amplitude_threshold:
            self.silence_counter += 1
        else:
            self.silence_counter = 0

        # Trigger prediction when user stops speaking
        if self.silence_counter >= self.silence_threshold and not self.predicted:
            self.predicted = True
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            sf.write(temp_file.name, np.concatenate(self.frames, axis=0), 44100)

            def run_pred():
                emotion, probs = predict_emotion(temp_file.name)
                st.session_state["live_emotion"] = emotion
                st.session_state["live_probs"] = probs

            threading.Thread(target=run_pred).start()

        return frame

webrtc_ctx = webrtc_streamer(
    key="live-audio",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=LiveAudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True
)

# Display live prediction results
if "live_emotion" in st.session_state:
    emotion = st.session_state["live_emotion"]
    probabilities = st.session_state["live_probs"]

    emotion_icons = {
        "neutral": "😐", "calm": "😌", "happy": "😄",
        "sad": "😢", "angry": "😡", "fearful": "😨",
        "disgust": "🤢", "surprised": "😲"
    }
    icon = emotion_icons.get(emotion, "❓")
    st.success(f"### Live Predicted Emotion: {icon} {emotion.capitalize()}")
    st.subheader("Prediction Confidence")
    st.bar_chart(dict(zip(le.classes_, probabilities)))
