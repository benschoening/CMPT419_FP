# CMPT419 Final Project
#
# File name: audio_helper.py
#
# Authors: Benjamin Schoening, Otto Mao
#
# Description: Deep Neural Network to classify audio files based on their extracted features

#imports
import torch
import torch.nn as nn
#/imports

#Audio DNN

class AudioDNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, n_classes, dropout):
        super(AudioDNN, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        dropped = self.dropout(last_hidden)
        fin = self.fc(dropped) # dropout
        return fin