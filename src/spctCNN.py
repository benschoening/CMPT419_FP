# CMPT419 Final Project
#
# File name: autoencoder.py
#
# Authors: Benjamin Schoening, Otto Mao
#
# Description: Spectrogram CNN

#imports
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

from audio_helper import AudioDataset

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#/imports

# WAV to SPEC
def spectrogram_wav(filepath, duration=5.0, sample_rate=22050, n_fft=2048, hop_length=512):
    target_length = int(sample_rate * duration)
    try:
        audio, sr = librosa.load(filepath, sr=sample_rate)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

    noise_profile = audio[:int(0.5 * sr)]
    audio = nr.reduce_noise(y=audio, sr=sr, y_noise=noise_profile)

    # Ensure audio is the correct length (pad or randomly crop)
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), mode="constant")
    else:
        max_start = len(audio) - target_length
        start_idx = random.randint(0, max_start)
        audio = audio[start_idx:start_idx + target_length]

    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(stft)

    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
    return spectrogram_db

# LOAD DATA
def load_dataset_spectrogram(data_dir, classes, duration=5.0, sample_rate=22050, n_fft=2048, hop_length=512):
    specs = []
    labels = []
    file_list = glob.glob(os.path.join(data_dir, "*/*.wav"))
    for filepath in file_list:
        label = os.path.basename(os.path.dirname(filepath))
        
        if label not in classes:
            continue
        
        spec = spectrogram_wav(filepath, duration, sample_rate, n_fft, hop_length)
        if spec is not None:
            specs.append(spec)

            label = os.path.basename(os.path.dirname(filepath))
            labels.append(label)
    specs = np.array(specs)
    labels = np.array(labels)

    specs = specs[:, np.newaxis, :, :]
    return specs, labels


# CNN
class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super(AudioCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #nn.Dropout(0.2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #nn.Dropout(0.2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #nn.Dropout(0.2)
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(128, num_classes),

        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# TRAIN
def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    model.train()    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0
        
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            epoch_accuracy = 100 * correct_preds / total_samples

            avg_loss = running_loss / 10
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}], Training Accuracy: {epoch_accuracy:.2f}%, Loss: {avg_loss:.4f}")
            running_loss = 0.0
        
        # epoch_loss = running_loss / len(dataloader)
        # epoch_accuracy = 100 * correct_preds / total_samples
        # print(f"Epoch [{epoch+1}/{num_epochs}] Training Accuracy: {epoch_accuracy:.2f}% Loss: {epoch_loss:.4f}")


# TRAINING AND TESTING
if __name__ == "__main__":

    data_dir = "data" 
    classes = ["Disgust", "Nervousness", "Smugness", "Uncertainty"]
    specs, labels = load_dataset_spectrogram(data_dir, classes, duration=5.0)

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    dataset_size = len(specs)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    split = int(0.8 * dataset_size)
    train_indices, test_indices = indices[:split], indices[split:]

    train_specs = specs[train_indices]
    train_labels = labels_encoded[train_indices]
    test_specs = specs[test_indices]
    test_labels = labels_encoded[test_indices]

    train_dataset = AudioDataset(train_specs, train_labels)
    test_dataset = AudioDataset(test_specs, test_labels)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    num_classes = len(label_encoder.classes_)
    model = AudioCNN(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00075)

    num_epochs = 10
    train_model(model, train_loader, criterion, optimizer, num_epochs, device)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    
# REPORT
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    class_names = label_encoder.classes_ 

    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig("results/cnn_confusion_matrix.png")
    
torch.save(model.state_dict(), "models/audio_cnn_weights.pth")