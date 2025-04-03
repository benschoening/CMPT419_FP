# CMPT419 Final Project
#
# File name: autoencoder.py
#
# Authors: Benjamin Schoening, Otto Mao
#
# Description: Autoencoder model Training and Evaluation

#imports
from audio_helper import *
from autoencoder import *
#/imports

#Data loader
data_dir = 'data'
mfcc, labels = load_dataset(data_dir, duration=10.0, sample_rate=22050, n_mfcc=20)




