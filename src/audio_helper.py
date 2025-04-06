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
import noisereduce as nr
import random
#/Imports

#MFCCs of audio files (with 20 features returned)
#No need to pre-process since using mfcc at the moment
def mfcc_wav(filepath, duration = 5.0, sample_rate = 22050, n_mfcc = 20):

    target_length = int(sample_rate * duration)

    #loads wav with exception e
    try:
        audio, sr = librosa.load(filepath, sr=sample_rate)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None
    
    #Using Noise Reduce to take out any background noise to enhance classification
    noise_profile = audio[:int(0.5 * sr)]  #first 0.5 second silent removed
    audio = nr.reduce_noise(y=audio, sr=sr, y_noise=noise_profile)
    
    #Target length acquired
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
    else:
        #audio = audio[:target_length] #First _ seconds of audio

        max_start = len(audio) - target_length #random selection of audio
        start_idx = random.randint(0, max_start)
        audio = audio[start_idx:start_idx + target_length]
    
    #outputs mfcc associated with sound
    mfcc = librosa.feature.mfcc(y=audio, n_mfcc=n_mfcc)

    #chroma, contrast plus combination
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    rms = librosa.feature.rms(y=audio)


    min_frames = min(mfcc.shape[1], chroma.shape[1], contrast.shape[1])
    mfcc = mfcc[:, :min_frames]
    chroma = chroma[:, :min_frames]
    contrast = contrast[:, :min_frames]
    rolloff = rolloff[:, :min_frames]
    centroid = centroid[:, :min_frames]
    zcr = zcr[:, :min_frames]
    rms = rms[:, :min_frames]

    combined = np.vstack([mfcc, chroma, contrast, rolloff, centroid, zcr, rms]).T
    #end 

    return combined #mfcc.T

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

#AUNTOENCODER DATASET
class AudioDataset1(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):

        return len(self.data)
    
    def __getitem__(self, idx):
        
        return self.data[idx], self.data[idx] #input and target, no labels returned for autoencoder ( FOR autoencoder vs DNN)

#DNN DATASAT
class AudioDataset2(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):

        return len(self.data)
    
    def __getitem__(self, idx):
        
        return self.data[idx], self.labels[idx]