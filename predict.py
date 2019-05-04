from keras.models import model_from_json
from keras.models import load_model
import load_data
from sklearn.preprocessing import normalize
import numpy as np
import keras
import keras_metrics
import IPython
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import ffmpeg
import librosa
import librosa.display
import time

print("Starting")
# load json and create model
json_file = open('model_config.json', 'r')

loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights("model_weights.h5")
print("Loaded model from disk")

def load_data_and_label():
    ## Sounds in which you can hear a bat are in the folder called "1". Others are in a folder called "0".
    batsounds = load_data.load_sounds_in_folder('./data/1')
    noisesounds = load_data.load_sounds_in_folder('./data/0')

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
    return indata, outdata, valin, valout


# indata, outdata, valin, valout = load_data_and_label()
#
# input = np.array(indata + valin)
# label = np.array(outdata + valout)
# model.compile(loss="binary_crossentropy", optimizer='sgd', metrics=[keras_metrics.precision(), keras_metrics.recall(),'accuracy'])
# valresults = model.evaluate(np.array(input), np.array(label), verbose=1)
# res_and_name = list(zip(valresults, model.metrics_names))
# for result,name in res_and_name:
#     print("Validation " + name + ": " + str(result))
#
# predicted = model.predict(np.array(input))
#
# for i in range(len(predicted)):
#     print("real :", label[i], ", predict:", predicted[i],
#           ",\t\t  Yes+" if predicted[i][1] > 0.5 and label[i][1] > 0.5 else "",
#           ",  T+" if predicted[i][1] > 0.5 else "",
#           ",  x" if predicted[i][1] > 0.2 else "")


print("Load long data")
soundarray, sr = librosa.load("data/test_data.m4a")
print("SR: ", sr)
maxseconds = int(len(soundarray)/sr)
for second in range(maxseconds-1):
    audiosample = np.array(soundarray[second*sr:int((second+.5)*sr)])
    metadata = load_data.audio_to_metadata(audiosample)
    testinput = normalize(np.array([metadata]),axis=1)
    prediction = model.predict(testinput)

    if np.argmax(prediction) ==1:
        IPython.display.display(IPython.display.Audio(audiosample, rate=sr,autoplay=True))
        time.sleep(2)
        print("Detected a bat at " + str(second) + " out of " + str(maxseconds) + " seconds")
        print(prediction)

    audiosample = np.array(soundarray[int((second + .5) * sr):int((second + 1) * sr)])
    metadata = load_data.audio_to_metadata(audiosample)
    testinput = normalize(np.array([metadata]), axis=1)
    prediction = model.predict(testinput)

    if np.argmax(prediction) ==1:
        IPython.display.display(IPython.display.Audio(audiosample, rate=sr,autoplay=True))
        time.sleep(2)
        print("Detected a bat at " + str(second) + " out of " + str(maxseconds) + " seconds")
        print(prediction)