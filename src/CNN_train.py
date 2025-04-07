# CMPT419 Final Project
#
# File name: autoencoder.py
#
# Authors: Benjamin Schoening, Otto Mao
#
# Description: Spectrogram CNN

#imports
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from CNN_helper import spectrogram_wav, AudioDataset, AudioCNN, load_dataset_spectrogram, train_model

#/imports

# TRAINING AND TESTING
if __name__ == "__main__":

    #print(torch.cuda.is_available())
    #print(torch.zeros(1).cuda())

    data_dir = "data" 
    classes = ["Disgust", "Nervousness", "Confidence", "Uncertainty"]
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
    test_dataset = AudioDataset (test_specs, test_labels)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    num_classes = len(label_encoder.classes_)
    model = AudioCNN(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00075   )

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
    
torch.save(model.state_dict(), "models/audio_cnn.pth")