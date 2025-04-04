# CMPT419 Final Project
#
# File name: audio_helper.py
#
# Authors: Benjamin Schoening, Otto Mao
#
# Description: Backend Functions for Extraction (and storage), Pre-processing, and Processing of Audio Files (.wav)


#Imports
import numpy as np
import librosa
import torch
import os
import glob
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
#/Imports

#MFCCs of audio files (with 20 features returned)
#No need to pre-process since using mfcc at the moment
def mfcc_wav(filepath, duration = 10.0, sample_rate = 22050, n_mfcc = 20):

    target_length = int(sample_rate * duration)

    #loads wav
    try:
        audio, sr = librosa.load(filepath, sr=sample_rate)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None
    
    #Target length acquired
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
    else:
        audio = audio[:target_length]
    
    #outputs mfcc associated with sound
    mfcc = librosa.feature.mfcc(y=audio, n_mfcc=n_mfcc)
    return mfcc.T

def load_dataset(data_dir, duration = 10.0, sample_rate=22050, n_mfcc=20):

    #mfcc = [] Stores MFCC from data directory
    #labels = [] Stores Label of given data based on 

    mfcc = []
    labels = []

    #
    file_list = glob.glob(os.path.join(data_dir, '*/*.wav'))

    for filepath in file_list:
        features = mfcc_wav(filepath, duration, sample_rate, n_mfcc) #loads wav, outputs mfcc
        if features is not None:
            mfcc.append(features)
            # Extract label from the parent folder name
            label = os.path.basename(os.path.dirname(filepath))
            labels.append(label)
    
    return np.array(mfcc), np.array(labels)


class AudioDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.data[idx] #input and target, no labels returned

