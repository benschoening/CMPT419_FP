import numpy as np
import librosa
import torch
import os
import glob
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn

from torch.utils.data import Dataset
import noisereduce as nr
import random

from sklearn.metrics import classification_report, confusion_matrix

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

# Dataset
class AudioDataset(Dataset):
    def __init__(self, specs, labels):
        self.data = torch.tensor(specs, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


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