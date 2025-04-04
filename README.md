# CMPT419_FP
Improving audio-text communication: Using AI models to classify social signals as emoji in short audio messages

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