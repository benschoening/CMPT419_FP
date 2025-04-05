# CMPT419_FP
Improving audio-text communication: Using AI models to classify social signals as emoji in short audio messages

- Goal to select audio social signals, categorize them into different categories (using a LSTM autoencoder), and have users interact and validate the classification based on the emoji presented

# Environment Setup:
conda create --name CMPT419_FP python=3.8
conda activate CMPT419_FP
pip install -r requirements.txt

# Directories
data/
    Uncertainty/
         unc_1.wav
         unc_2.wav
         ...
    Nervousness/
         nerv_1.wav
         nerv_2.wav
         ...
    Disgust/
         dis_1.wav
         dis_2.wav
         ...
    Smugness/
         smug_1.wav
         smug_2.wav
         ...

src/
     audio_helper.py

requirements.txt

README.md

dataset_key.pdf



# Data Description:
Refer to dataset_key.pdf

# Challenges
1. Data availability: finding and collecting data proved to be challenging, by the amount of time to find specific examples of each social signal. One example of this challenge is not being able to find enough data for 'smugness', where we had to change our social signal to a more broader signal of confidence
2. Classification: had to use different hyperparameters, models, data, and audio techniques to find best result


# Sources
https://medium.com/@joshiprerak123/transform-your-audio-denoise-and-enhance-sound-quality-with-python-using-pedalboard-24da7c1df042
https://pythonguides.com/python-tkinter-image/


# Potential ideas
- add chroma features (chroma = librosa.feature.chroma_stft(y=audio, sr=sr))
- add spectral contrast (contrast = librosa.feature.spectral_contrast(y=audio, sr=sr))
- add zero crossing rate (zcr = librosa.feature.zero_crossing_rate(audio))
- add deltas of MFCC (delta_mfcc = librosa.feature.delta(mfcc), delta2_mfcc = librosa.feature.delta(mfcc, order=2)))
