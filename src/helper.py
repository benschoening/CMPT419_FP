# CMPT419 Final Project
#
# File name: helper.py
#
# Authors: Benjamin Schoening, Otto Mao
#
# Description: Backend Functions for Extraction (and storage), Pre-processing, and Processing of Audio Files (.wav)


#Imports
import numpy as np
import librosa
import os
#/Imports

def load_wav(filepath, duration = 10.0, sample_rate = 22050, n_mfcc = 20):

    target_length = int(sample_rate * duration)

    #loads wav, and outputs mfcc associated with sound
    try:
        audio, sr = librosa.load(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None
    
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
    else:
        audio = audio[:target_length]
    
    mfcc = librosa.feature.mfcc(y=audio, n_mfcc=n_mfcc)

    return mfcc.T