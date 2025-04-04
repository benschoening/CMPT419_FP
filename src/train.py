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
import matplotlib.pyplot as plt
#/imports

#Data loader
data_dir = 'data'
mfcc, labels = load_dataset(data_dir, duration=5.0, sample_rate=22050, n_mfcc=20)


#Encoding string labels to int vals
le = LabelEncoder()
int_labels = le.fit_transform(labels)

dataset = AudioDataset(mfcc, int_labels)

dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

#Data Augmentation??????

#-----------------------

#Hyper parameters
learning_rate = 0.01
epoch = 25
n_mfcc = 20
seq_len = 216  #dependent on duration
hidden_dim = 64
latent_dim = 30
num_layers = 3


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


#Visualization and Evaluation (Scatter Plot)

model.eval()
latent_list = []

with torch.no_grad():
    for inputs, labels in dataloader:
        inputs = inputs.to(device)

        _, latent = model(inputs) #encoder latent representations
        latent_list.append(latent.cpu().numpy()) #np arrays
latents = np.concatenate(latent_list, axis=0)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(latents[:, 0], latents[:, 1], c=int_labels, cmap='viridis', alpha=0.7)
plt.colorbar(scatter)
plt.title(f"Latent Space at Epoch {epoch}")
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.show()



