# CMPT419_FP
Improving audio-text communication: Using AI models to classify social signals as emoji in short audio messages

- Goal to select audio social signals, categorize them into different categories (using a LSTM autoencoder), and have users interact and validate the classification based on the emoji presented

## Environment Setup:
1. '''conda create --name CMPT419_FP python=3.8'''
2. '''conda activate CMPT419_FP'''
3. '''pip install -r requirements.txt'''

### Usage
To run the project, use the following commands to show results based on :

- ''' python src/UI.py --autoencoder'''
- ''' python src/UI.py --DNN'''
- ''' python src/UI.py --CNN'''

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
-     Other/
          oth_1.wav,
          oth_2.wav,
          ...

-    results/
- src/
     audio_helper.py, autoencoder.py, DNN.py, train.py, UI.py

- requirements.txt

### Models
1. LTSM Autoencoder
2. LTSM Deep Neural Network
3. CNN (spectrograms)


### Data Description:
Dataset key here

### Challenges
1. Data availability: finding and collecting data proved to be challenging, by the amount of time to find specific examples of each social signal. One example of this challenge is not being able to find enough data for 'smugness', where we had to change our social signal to a more broader signal of confidence
2. Classification: had to use different hyperparameters, models, data, and audio techniques to find best result

### Changes made during project
- Social Signal of 'smugness' to 'confidence'
- implementation of 3 different models to find most effective classification

# Sources
https://medium.com/@joshiprerak123/transform-your-audio-denoise-and-enhance-sound-quality-with-python-using-pedalboard-24da7c1df042
https://pythonguides.com/python-tkinter-image/


