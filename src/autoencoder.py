# CMPT419 Final Project
#
# File name: autoencoder.py
#
# Authors: Benjamin Schoening, Otto Mao
#
# Description: Autoencoder model architecture (might be some more implementations)

#imports
import torch
import torch.nn as nn
from audio_helper import *
#/imports

#Autoencoder using LTSM nn
class LTSM_autoencoder(nn.Module):
    def __init__(self, n_mfcc, seq_len, hidden_dim=64, latent_dim=30, num_layers=1):

        super(LTSM_autoencoder, self).__init__()

        class Encoder(nn.Module):
            def __init__(self, n_mfcc, hidden_dim, latent_dim, num_layers):
                super(Encoder, self).__init__()

                #LTSM layer
                self.ltsm = nn.LSTM(input_size=n_mfcc, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

                #fully connected layer
                self.e_fc = nn.Linear(hidden_dim, latent_dim)
            def forward(self, x):
                _, (hn, _) = self.ltsm(x) #(num_layers, batch, hidden_dim)
                h = hn[-1] #last (batch, hidden_dim)
                latent = self.e_fc(h)
                return latent #compressed audio data
        
        class Decoder(nn.Module):
            def __init__(self, latent_dim, hidden_dim, n_mfcc, num_layers, seq_len):
                super(Decoder, self).__init__()

                self.fc = nn.Linear(latent_dim, hidden_dim) #decompress back to hidden_dim

                self.seq_len = seq_len

                self.ltsm = nn.LSTM(input_size=hidden_dim, hidden_size=n_mfcc, num_layers=num_layers, batch_first=True)

                self.activation = nn.Sigmoid()

            def forward(self, latent):
                dec = self.fc(latent)
                dec_repeat = dec.unsqueeze(1).repeat(1, self.seq_len, 1)

                dec_out, _ = (self.ltsm(dec_repeat))
                return torch.sigmoid(dec_out) #(bach, seq_len, n_mfcc)

        self.encoder = Encoder(n_mfcc, hidden_dim, latent_dim, num_layers)
        self.decoder = Decoder(latent_dim, hidden_dim, n_mfcc, num_layers, seq_len)

    def forward(self, x):
        latent = self.encoder(x)
        out = self.decoder(latent)

        return out, latent



        