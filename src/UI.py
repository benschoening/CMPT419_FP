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
#/imports

window = tk.Tk()

greeting = tk.Label(text="Welcome to the interactive ")
greeting.pack(pady=100)

img = Image.open('src/imgs/disgust_emoji.jpg')  
img = ImageTk.PhotoImage(img)

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


audio_options = []
for root, dirs, files in os.walk('data'):
    for file in files:
        if file.lower().endswith('.wav'): 
            audio_options.append(os.path.join(root, file))

audio_options.sort(key=natural_sort_key)

audio_var = tk.StringVar(window)
audio_var.set(audio_options[0])

audio_combobox = ttk.Combobox(window, values=audio_options, font=("Helvetica", 12), textvariable=audio_var)
audio_combobox.config(height=15)
audio_combobox.pack(pady=10)

def play_audio():
    selected_audio = audio_var.get()
    winsound.PlaySound(selected_audio, winsound.SND_FILENAME | winsound.SND_ASYNC)

play_button = tk.Button(window, text="Play Audio", font=("Helvetica", 32), command=play_audio)
play_button.pack(pady=20)

window.mainloop()
