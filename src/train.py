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
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
#/imports

#Data loader
data_dir = 'data'
mfcc, labels = load_dataset(data_dir, duration=5.0, sample_rate=22050, n_mfcc=20)


#Normalization of data
all_mfcc = np.concatenate(mfcc, axis=0)
scaler = StandardScaler().fit(all_mfcc)

#scalar to each sequence
mfcc_normalized = [scaler.transform(seq) for seq in mfcc]


#Encoding string labels to int vals
le = LabelEncoder()
int_labels = le.fit_transform(labels)

dataset = AudioDataset(mfcc_normalized, int_labels)

dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

#Hyper parameters
learning_rate = 0.001
epoch = 10
n_mfcc = 43 #32 with chroma, 39 with contrast + chroma, 43 for all
seq_len = 216  #dependent on duration
hidden_dim = 64
latent_dim = 16
num_layers = 2


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

#K-means and scores
kmeans = KMeans(n_clusters=4)
cluster_labels = kmeans.fit_predict(latents)

ari = adjusted_rand_score(int_labels, cluster_labels)
sil_score = silhouette_score(latents, cluster_labels)

print(f"Adjusted Rand Index: {ari:.3f}")
print(f"Silhouette Score: {sil_score:.3f}")



fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns

#Cluster labels from K-Means
scatter1 = axes[0].scatter(latents[:, 0], latents[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
axes[0].set_title(f"K-Means Clustering (Epoch {epoch})")
axes[0].set_xlabel("Latent Dimension 1")
axes[0].set_ylabel("Latent Dimension 2")
fig.colorbar(scatter1, ax=axes[0], label="Cluster ID")

#True labels
scatter2 = axes[1].scatter(latents[:, 0], latents[:, 1], c=int_labels, cmap='viridis', alpha=0.7)
axes[1].set_title(f"True Labels (Epoch {epoch})")
axes[1].set_xlabel("Latent Dimension 1")
axes[1].set_ylabel("Latent Dimension 2")
fig.colorbar(scatter2, ax=axes[1], label="True Label")

plt.tight_layout()
plt.show()


