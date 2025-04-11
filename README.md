# CMPT419_FP
Improving audio-text communication: Using AI models to classify social signals as emoji in short audio messages

- Goal to select audio social signals, categorize them into different categories (using a LSTM autoencoder), and have users interact and validate the classification based on the emoji presented

## Environment Setup:
1. '''conda create --name CMPT419_FP python=3.8'''
2. '''conda activate CMPT419_FP'''
3. '''pip install -r requirements.txt'''

### Models
1. LSTM Autoencoder
2. LSTM DNN
3. Spectrogram CNN

### Usage
To run the User Interactive aspect of this project, use the following commands to show results based on :

- "python src/UI.py"

To run and see the results of each individual model use the commands:

For LSTM DNN and LSTM Autoencoder use:
- "python src/train.py 

For Spectrogram CNN use:
- "python src/CNN_train.py

### Directories
data/
-    Uncertainty/
         unc_1.wav,
         unc_2.wav,
         ...
-    Nervousness/
         nerv_1.wav,
         nerv_2.wav,
         ...
-    Disgust/
         dis_1.wav,
         dis_2.wav,
         ...
-    Confidence/
         con_1.wav,
         con_2.wav,
         ...
-    Other/
          oth_1.wav,
          oth_2.wav,
          ...

-    results/
          photos
-     src/
     audio_helper.py, autoencoder.py, DNN.py, train.py, UI.py, CNN_helper.py, CNN_train.py

- requirements.txt


### Data Description:
Provided in dataset_key.pdf

### Challenges
1. Data availability: finding and collecting data proved to be challenging, by the amount of time to find specific examples of each social signal. One example of this challenge is not being able to find enough data for 'smugness', where we had to change our social signal to a more broader signal of confidence
2. Classification: had to use different hyperparameters, models, data, and audio techniques to find best result

### Changes made during project
- Social Signal of 'smugness' to 'confidence'
- implementation of 3 different models to find most effective classification

# Sources
1. https://medium.com/@joshiprerak123/transform-your-audio-denoise-and-enhance-sound-quality-with-python-using-pedalboard-24da7c1df042
https://pythonguides.com/python-tkinter-image/
2. https://scikit-learn.org/stable/index.html
3. https://towardsdatascience.com/what-is-stratified-cross-validation-in-machine-learning-8844f3e7ae8e/
4. https://librosa.org/doc/main/feature.html



