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
#/imports

window = tk.Tk()

greeting = tk.Label(text="Welcome to the interactive ")
greeting.pack(pady=100)

img = Image.open('src/imgs/disgust_emoji.jpg')  
img = ImageTk.PhotoImage(img)

play_button = tk.Button(window, text="Play Audio", font=("Helvetica", 32)) #enter commands to play audio
play_button.pack(pady=20)



window.mainloop()
