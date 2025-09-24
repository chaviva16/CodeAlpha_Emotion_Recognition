## Emotion Recognition App

Emotion Recognition is a web app that detects the emotion conveyed in uploaded speech audio files. It uses a deep learning model (CNN) trained with RAVDESS, TESS, and CREMA-D datasets. The model was further improved with data augmentation and class balancing, leading to strong performance in recognizing emotions like Neutral, Happy, Sad, Angry, Fearful, Disgust, and Surprised.

👉 Live Demo: https://voicemood.streamlit.app/

## Features

Emotion Detection – Recognizes one of 7 emotions from speech audio.

Data Augmentation – Robustness improved via noise, pitch, and speed transformations.

Class Balancing – Prevents bias toward overrepresented emotions.

 Prediction Confidence – Displays probability scores for each emotion.

Fast Processing – Optimized for 3–4s audio clips to reduce lag.

 Clean UI – Intuitive design with emojis and file upload.


## Model Development Journey

Baseline Model (RAVDESS only)

Dataset: RAVDESS (Actors 01–24, 8 emotions)

Architecture: Basic CNN with MFCC features (40 coefficients).

Accuracy: ~74%

Weaknesses: Struggled with overlapping classes (Happy ↔ Neutral, Calm ↔ Sad).


## Improvements

 Added data augmentation (noise, pitch, speed, shift).

 Applied class weighting to balance minority emotions.

Helped reduce bias but accuracy stayed ~74%.

Extended Training (Multi-dataset)

Added TESS and CREMA-D datasets.

Now trained on thousands more samples → much more robust.

Final Accuracy: ~81% 


##  Model Evaluation
Confusion Matrix Observations

High performance on Angry, Disgust, Fearful, Surprised.

⚠️ Overlaps remain:

Happy ↔ Neutral → model sometimes confuses cheerful voices with neutral ones.

Calm ↔ Sad → calm tones misread as sadness.

This reflects a real-world challenge: subtle emotions are difficult even for humans.


## Challenges & Limitations

Dataset is still relatively small compared to large-scale benchmarks.

Predictions rely only on audio (no facial or text cues).

Longer audios are clipped at 4s → might miss late emotional cues.


## Future Work

Try CNN + BiLSTM or transformer-based approaches (e.g., wav2vec).

Use richer features (spectral contrast, chroma, zero-crossing rate).

Train on larger, more varied emotion datasets.

Add batch testing mode for multiple files at once.


##  Dataset Sources

RAVDESS – Ryerson Audio-Visual Database of Emotional Speech and Song

 TESS – Toronto Emotional Speech Set

 CREMA-D – Crowd-Sourced Emotional Multimodal Actors Dataset


## Tech Stack

Frameworks: TensorFlow / Keras, librosa, Streamlit

Features: MFCC (40 coefficients), padded/truncated to fixed length

Deployment: Streamlit Cloud


## Results

Initial RAVDESS-only model: ~64% accuracy

Improved with augmentation + class balancing: ~74%

Final CNN with RAVDESS + TESS + CREMA-D: ~81% accuracy 


## Credits

Datasets: RAVDESS, TESS, CREMA-D

Libraries: TensorFlow, Keras, librosa, Streamlit

Special thanks to the open source community 💙.
