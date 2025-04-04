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
from sklearn.preprocessing import LabelEncoder
import torch.optim as optim
#/imports

#Data loader
data_dir = 'data'
mfcc, labels = load_dataset(data_dir, duration=10.0, sample_rate=22050, n_mfcc=20)


#Encoding string labels to int vals
le = LabelEncoder()
int_labels = le.fit_transform(labels)

dataset = AudioDataset(mfcc)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

#Data Augmentation??????

#-----------------------

#Hyper parameters
learning_rate = 0.001
epoch = 5
n_mfcc = 20
seq_len = 431  #dependent on duration
hidden_dim = 64
latent_dim = 30
num_layers = 1


#Create model
model = LTSM_autoencoder(n_mfcc, seq_len, hidden_dim, latent_dim, num_layers)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

#Training portion
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


for i in range(epoch):
    model.train()
    running_loss = 0.0

    for inputs, target in dataloader:
        inputs = inputs.to(device)
        optimizer.zero_grad()

        #forward pass
        outputs, latent = model(inputs)
        #reconstruction loss
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {i+1}/{epoch} - Loss: {epoch_loss:.4f}")




