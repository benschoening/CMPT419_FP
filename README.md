# CMPT419_FP
Improving audio-text communication: Using AI models to classify feelings as emoji in short audio messages

# If needed to change python in env
https://www.askpython.com/python/examples/change-the-python-version-conda


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
