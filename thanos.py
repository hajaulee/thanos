import random
import sys
import glob
import os
import time

import IPython
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram

import librosa
import librosa.display

from sklearn.preprocessing import normalize
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import keras_metrics
import load_data

## Sounds in which you can hear a bat are in the folder called "1". Others are in a folder called "0".
batsounds = load_data.load_sounds_in_folder('./data/1')
noisesounds = load_data.load_sounds_in_folder('./data/0')

print("With bat: %d without: %d total: %d " % (len(batsounds), len(noisesounds), len(batsounds)+len(noisesounds)))
print("Example of a sound with a bat:")
IPython.display.display(IPython.display.Audio(random.choice(batsounds), rate=load_data.SR,autoplay=True))
print("Example of a sound without a bat:")
IPython.display.display(IPython.display.Audio(random.choice(noisesounds), rate=load_data.SR,autoplay=True))



def get_short_time_fourier_transform(soundwave):
    return librosa.stft(soundwave, n_fft=256)

def short_time_fourier_transform_amplitude_to_db(stft):
    return librosa.amplitude_to_db(stft)

def soundwave_to_np_spectogram(soundwave):
    step1 = get_short_time_fourier_transform(soundwave)
    step2 = short_time_fourier_transform_amplitude_to_db(step1)
    step3 = step2/100
    return step3

def inspect_data(sound):
    plt.figure()
    plt.plot(sound)
    IPython.display.display(IPython.display.Audio(sound, rate=load_data.SR))
    a = get_short_time_fourier_transform(sound)
    Xdb = short_time_fourier_transform_amplitude_to_db(a)
    plt.figure()
    plt.imshow(Xdb)
    plt.show()
    print("Length per sample: %d, shape of spectogram: %s, max: %f min: %f" % (len(sound), str(Xdb.shape), Xdb.max(), Xdb.min()))

inspect_data(batsounds[0])
inspect_data(noisesounds[0])



metadata = load_data.audio_to_metadata(batsounds[0])
print(metadata)
print(len(metadata))



# Meta-feature based batsounds and their labels
preprocessed_batsounds = list()
preprocessed_noisesounds = list()

for sound in batsounds:
    expandedsound = load_data.audio_to_metadata(sound)
    preprocessed_batsounds.append(expandedsound)
for sound in noisesounds:
    expandedsound = load_data.audio_to_metadata(sound)
    preprocessed_noisesounds.append(expandedsound)

labels = [0]*len(preprocessed_noisesounds) + [1]*len(preprocessed_batsounds)
assert len(labels) == len(preprocessed_noisesounds) + len(preprocessed_batsounds)
allsounds = preprocessed_noisesounds + preprocessed_batsounds
allsounds_normalized = normalize(np.array(allsounds),axis=1)
one_hot_labels = keras.utils.to_categorical(labels)
print(allsounds_normalized.shape)
print("Total noise: %d total bat: %d total: %d" % (len(allsounds_normalized), len(preprocessed_batsounds), len(allsounds)))

## Now zip the sounds and labels, shuffle them, and split into a train and testdataset
zipped_data = list(zip(allsounds_normalized, one_hot_labels))
np.random.shuffle(zipped_data)
random_zipped_data = zipped_data
VALIDATION_PERCENT = 0.8 # use X percent for training, the rest for validation
traindata = random_zipped_data[0:int(VALIDATION_PERCENT*len(random_zipped_data))]
valdata = random_zipped_data[int(VALIDATION_PERCENT*len(random_zipped_data))::]
indata = [x[0] for x in traindata]
outdata = [x[1] for x in traindata]
valin = [x[0] for x in valdata]
valout = [x[1] for x in valdata]


LEN_SOUND = len(preprocessed_batsounds[0])
NUM_CLASSES = 2 # Bat or no bat

print("InputLEN:", LEN_SOUND)
from keras.callbacks import ModelCheckpoint
output_filename = 'model.json'
# callbacks
checkpointer = ModelCheckpoint(
    filepath=output_filename,
    save_best_only=True
)

model = Sequential()
model.add(Dense(131, activation='relu',input_shape=(LEN_SOUND,)))
model.add(Dense(97, activation='relu'))
model.add(Dense(31, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss="binary_crossentropy", optimizer='sgd', metrics=['accuracy'])
model.summary()
model.fit(np.array(indata),
          np.array(outdata),
          batch_size=64,
          epochs=2000,
          verbose=2,
          validation_data=(np.array(valin), np.array(valout)),
          shuffle=True,
          callbacks=[checkpointer])
valresults = model.evaluate(np.array(valin), np.array(valout), verbose=0)
res_and_name = list(zip(valresults, model.metrics_names))
for result,name in res_and_name:
    print("Validation " + name + ": " + str(result))

predicted = model.predict(np.array(valin))

for i in range(len(predicted)):
    print("real :", valout[i], ", predict:", predicted[i], ",  T+" if predicted[i][1] > 0.5 else "", ",  x" if predicted[i][1] > 0.2 else "")


model_json = model.to_json()  # save just the config. replace with "to_yaml" for YAML serialization
with open("model_config.json", "w") as f:
    f.write(model_json)

model.save_weights('model_weights.h5') # save just the weights.

model.save("fullModel.h5")


