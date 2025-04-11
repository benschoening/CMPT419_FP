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
from DNN import *
from sklearn.preprocessing import LabelEncoder
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
#/imports

#-----------------------Data loader for autoencoder--------------------------------------
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

dataset = AudioDataset1(mfcc_normalized, int_labels)

dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

#-----------LTSM autoencoder (hyper parameters embedded)------------------------
def train_autoencoder():

    #Hyper parameters (plus batch size above)
    learning_rate = 0.001
    epoch = 15
    num_feats = 43 #32 with chroma, 39 with contrast + chroma, 43 for all (displayed as n_mfcc in other files)
    seq_len = 216  #dependent on duration
    hidden_dim = 64
    latent_dim = 16
    num_layers = 2


    #Create model
    model = LTSM_autoencoder(num_feats, seq_len, hidden_dim, latent_dim, num_layers)

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
    
    #cm = confusion_matrix(int_labels, cluster_labels)

    # # Plot the confusion matrix heatmap.
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    # plt.xlabel("Predicted Cluster")
    # plt.ylabel("True Label")
    # plt.title("Confusion Matrix")
    # plt.savefig("results/autoencoder_confusion_matrix")
    
    torch.save(model.state_dict(), "models/audio_autoencoder.pth")
    
#-------------------------------------------------------------------------------

#------------------------------Data loader for DNN ------------------------------------------
data_dir = 'data'
feats, labels = load_dataset(data_dir, duration=5.0, sample_rate=22050, n_mfcc=20)

#Normalization of data
all_mfcc = np.concatenate(mfcc, axis=0)
scaler = StandardScaler().fit(all_mfcc)

#scalar to each sequence
feats_normalized = [scaler.transform(seq) for seq in mfcc]


#Encoding string labels to int vals
le = LabelEncoder()
int_labels = le.fit_transform(labels) 

#Train-Split
X_train, X_val, y_train, y_val = train_test_split(feats_normalized, int_labels, test_size=0.2, stratify=int_labels)

#DataLoaders
train_dataset = AudioDataset2(X_train, y_train)
test_dataset = AudioDataset2(X_val, y_val)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)


#----------------LTSM DNN (hyper parameters embedded)--------------------------
def train_DNN():

    #Hyper parameters
    input_dim = 43 #number of features extracted from audio
    hidden_dim = 128
    num_layers = 4
    n_classes = 5
    learning_rate = 0.0010
    epochs = 50
    dropout = 0.2

    #model creation + GPU
    model = AudioDNN(input_dim, hidden_dim, num_layers, n_classes, dropout)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) #regularization can be added ', weight_decay=1e-5'


    for i in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)

            prediction = torch.argmax(outputs, dim=1)
            correct += (prediction == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total
        print(f"Epoch {i+1}/{epochs} - Loss: {total_loss/total:.4f} - Accuracy: {accuracy:.4f}")

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    #Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    class_names = le.classes_

    print(classification_report(all_labels, all_preds, target_names=class_names))

    #CONFUSION MATRIX
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig("results/DNN_confusion_matrix.png")
    
    torch.save(model.state_dict(), "models/audio_dnn.pth")


#-------------------------------------------------------------------------------


#train_autoencoder()

train_DNN()