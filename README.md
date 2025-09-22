## Emotion Recognition App

Emotion Recognition is a web app that detects the emotion conveyed in uploaded speech audio files.
It uses a deep learning model (CNN) trained with the RAVDESS dataset and enriched with data augmentation & class weighting to improve performance and robustness. 
Simply upload a .wav file and let the app tell you what emotion it hears — Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, or Surprised.

Live Demo ( https://voicemood.streamlit.app/ )

##  Features

Emotion Detection	Recognizes one of 8 emotions from speech audio.
Data Augmentation	Boosted robustness via noise/pitch/speed changes during training.
Class Balancing	Gives more weight to underrepresented emotions.
Prediction Confidence	Shows probability scores for each emotion.
Fast Processing	Clips audio to 4 seconds, uses optimized loading to prevent lag.
Clean UI	Intuitive design with emojis and easy file upload.

## Model Details

Dataset: RAVDESS (Actors 01–24, 8 emotions)

Input features: MFCC (40 coefficients) with fixed time dimension (padding/truncate)

Model architecture: Convolutional Neural Network (CNN) with augmentation & class weighting

Final test accuracy: ~75%

Weak emotions: Neutral & Happy show more confusion due to overlapping audio features

## Challenges & Limitations

The dataset is somewhat small; some emotions such as Neutral and Happy tend to be more confused by the model.

Predictions are based purely on audio; no visual cues (face, gesture) or text tone are considered.

If the uploaded audio is longer than 4 seconds, the app only analyzes the first 4 seconds. This might occasionally cut off emotional cues.

## Future Work

Experiment with CNN + BiLSTM / transfer learning (e.g., wav2vec) to improve performance.

Add more audio features: e.g., spectral contrast, chroma, zero-crossing rate.

Expand model training on bigger or more varied datasets for better generalization.

Add a batch testing mode to allow evaluation of multiple files at once.

## Credits

Dataset: RAVDESS

Thanks to open-source libraries: TensorFlow / Keras, librosa, streamlit
