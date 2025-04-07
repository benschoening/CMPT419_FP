# CMPT419 Final Project
#
# File name: UI.py
#
# Authors: Benjamin Schoening, Otto Mao
#
# Description: User Interface for interaction and surveying of correct audio signals

#imports
import tkinter as tk
from PIL import Image, ImageTk
import os
import winsound
import re
from tkinter import ttk

from CNN_helper import AudioCNN, spectrogram_wav
import torch
import numpy as np

import sounddevice as sd
import soundfile as sf
import time

#/imports

# SETUP
window = tk.Tk()
window.title("Audio Player and Classification")

classes = ["Confidence", "Disgust", "Nervousness",  "Other", "Uncertainty"]

emoji_map = {
    "Confidence": "src/imgs/confident_emoji.png",
    "Uncertainty": "src/imgs/uncertain_emoji.png",
    "Nervousness": "src/imgs/nervous_emoji.png",
    "Other": "src/imgs/other_emoji.png",
    "Disgust": "src/imgs/disgust_emoji.jpg"
}

preloaded_emojis = {}
for label, path in emoji_map.items():
    if os.path.exists(path):
        img = Image.open(path)
        img = img.resize((100, 100))
        preloaded_emojis[label] = ImageTk.PhotoImage(img)

def load_model(model_path):
    num_classes = len(classes)
    model = AudioCNN(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

# DEFAULT MODEL
current_model = load_model("models/audio_cnn_67.pth")

left_frame = tk.Frame(window)
left_frame.grid(row=0, column=0, padx=20, pady=20)

title_label = tk.Label(left_frame, text="Prediction", font=("Helvetica", 16))
title_label.pack(pady=10)

current_prediction = tk.StringVar()
current_prediction.set("Prediction: None")

right_frame = tk.Frame(window)
right_frame.grid(row=0, column=1, padx=20, pady=20, sticky="n")

label = tk.Label(left_frame, image=preloaded_emojis.get("Other"))
label.image = preloaded_emojis.get("Other")  # keep reference
label.pack(pady=10)

prediction_text = tk.Label(left_frame, textvariable=current_prediction, font=("Helvetica", 14))
prediction_text.pack(pady=10)

# LEFT SIDE: PREDICTING PART
def predict_audio():
    selected_audio = audio_var.get()
    spec = spectrogram_wav(selected_audio, duration=5.0)
    
    if spec is None:
        print("Could not process audio.")
        return

    spec = np.expand_dims(spec, axis=0)
    spec = np.expand_dims(spec, axis=0)
    spec_tensor = torch.tensor(spec, dtype=torch.float32)

    with torch.no_grad():
        output = current_model(spec_tensor)
        #predicted_index = torch.argmax(output, dim=1).item()
        # for other
        probabilities = torch.softmax(output, dim=1)
        max, predicted_index = torch.max(probabilities, dim=1)
    
    #print("Confidence: ", max.item())
    
    confidence_threshold = 0.7
    if max.item() < confidence_threshold:
        predicted_label = "Other"
    else:
        predicted_label = classes[predicted_index.item()]
        
    predicted_label = classes[predicted_index]
    print("Predicted Label:", predicted_label)
    
    # Update the displayed emoji image based on the predicted label
    if predicted_label in preloaded_emojis:
        label.configure(image=preloaded_emojis[predicted_label])
        label.image = preloaded_emojis[predicted_label]
    else:
        print("Emoji image not preloaded for label:", predicted_label)
    current_prediction.set(f"Prediction: {predicted_label}")
        
predict_button = tk.Button(left_frame, text="Predict Audio", font=("Helvetica", 20), command=predict_audio)
predict_button.pack(pady=10)

# RIGHT SIDE: AUDIO SELECT
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split("([0-9]+)", s)]

audio_options = []
for root, dirs, files in os.walk("data"):
    for file in files:
        if file.lower().endswith(".wav"): 
            audio_options.append(os.path.join(root, file))

audio_options.sort(key=natural_sort_key)

audio_var = tk.StringVar(right_frame)
audio_var.set(audio_options[0])

audio_label = tk.Label(right_frame, text="Select Audio Clip", font=("Helvetica", 14))
audio_label.pack(pady=5)

audio_combobox = ttk.Combobox(right_frame, values=audio_options, font=("Helvetica", 14), textvariable=audio_var, width=40)
audio_combobox.config(height=15)
audio_combobox.pack(pady=5)

def play_audio():
    selected_audio = audio_var.get()
    winsound.PlaySound(selected_audio, winsound.SND_FILENAME | winsound.SND_ASYNC)

play_button = tk.Button(right_frame, text="Play Audio", font=("Helvetica", 14), command=play_audio)
play_button.pack(pady=10)

# MODEL SELECT

model_options = []
for root, dirs, files in os.walk("models"):
    for file in files:
        if file.lower().endswith(".pth"): 
            model_options.append(os.path.join(root, file))

model_var = tk.StringVar(right_frame)
model_var.set(model_options[0])

model_combobox = ttk.Combobox(right_frame, values=model_options, font=("Helvetica", 12), textvariable=model_var)
model_combobox.config(height=5)
model_combobox.pack(pady=10)

def load_selected_model():
    global current_model
    model_path = model_var.get()
    current_model = load_model(model_path)
    print("Loaded model:", model_path)

load_model_button = tk.Button(right_frame, text="Load Model", font=("Helvetica", 14), command=load_selected_model)
load_model_button.pack(pady=10)

recording = False
record_stream = None
file_writer = None
recorded_file = None
samplerate = 22050
channels = 1

def record_callback(indata, frames, time_info, status):
    if status:
        print(status)
    global file_writer
    if file_writer:
        file_writer.write(indata)

def start_recording():
    global recording, record_stream, file_writer, recorded_file
    recording = True
    recorded_folder = os.path.join("data", "Recorded")
    os.makedirs(recorded_folder, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    recorded_file = os.path.join(recorded_folder, f"recorded_{timestamp}.wav")
    # Open a sound file in write mode.
    file_writer = sf.SoundFile(recorded_file, mode='w', samplerate=samplerate, channels=channels)
    # Open the input stream with the callback.
    record_stream = sd.InputStream(samplerate=samplerate, channels=channels, callback=record_callback)
    record_stream.start()
    #print("Recording started:", recorded_file)
    
def normalize_audio(file_path):
    data, sr = sf.read(file_path)
    # Compute the maximum absolute amplitude.
    peak = np.max(np.abs(data))

    if peak == 0:
        return

    scaling_factor = 0.99 / peak
    normalized_data = data * scaling_factor
    # Write the normalized audio back to file.
    sf.write(file_path, normalized_data, sr)

    
def stop_recording():
    global recording, record_stream, file_writer
    if record_stream is not None:
        record_stream.stop()
        record_stream.close()
    if file_writer is not None:
        file_writer.close()
    recording = False

    normalize_audio(recorded_file)

    audio_options.insert(0, recorded_file)
    #audio_options.sort(key=natural_sort_key)
    audio_combobox['values'] = audio_options

def toggle_record():
    if not recording:
        record_button.config(text="Stop Recording")
        start_recording()
    else:
        stop_recording()
        record_button.config(text="Record Audio")
        
record_button = tk.Button(right_frame, text="Record Audio", font=("Helvetica", 14), command=toggle_record)
record_button.pack(pady=20, anchor="se")

window.mainloop()
