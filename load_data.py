import random
import sys
import glob
import os
import time
import librosa
import numpy as np
# Note: SR stands for sampling rate, the rate at which my audio files were recorded and saved.
SR = 22050 # All audio files are saved like this

def load_sounds_in_folder(foldername):
    """ Loads all sounds in a folder"""
    sounds = []
    for filename in os.listdir(foldername):
        X, sr = librosa.load(os.path.join(foldername,filename))
        if sr != SR: print("Other sr:", sr)
        assert sr == SR
        sounds.append(X)
    return sounds

WINDOW_WIDTH = 10
AUDIO_WINDOW_WIDTH = 1000 # With sampling rate of 22050 we get 22 samples for our second of audio
def audio_to_metadata(audio):
    """ Takes windows of audio data, per window it takes the max value, min value, mean and stdev values"""
    features = []
    for start in range(0,len(audio)-AUDIO_WINDOW_WIDTH,AUDIO_WINDOW_WIDTH):
        subpart = audio[start:start+AUDIO_WINDOW_WIDTH]
        maxval = max(subpart)
        minval = min(subpart)
        mean = np.mean(subpart)
        stdev = np.std(subpart)
        features.extend([maxval,minval,mean,stdev,maxval-minval])
    return features