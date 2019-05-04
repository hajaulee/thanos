import os
import pyaudio
import wave
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
import pyautogui
import tray_icon
import autopy

print("Starting")
# load json and create model
json_file = open('model_config.json', 'r')

loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights("model_weights.h5")
print("Loaded model from disk")

SR = 22050  # All audio files are saved like this
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = SR
CHUNK = 1024
RECORD_SECONDS = 0.5
WAVE_OUTPUT_FILENAME = "file.wav"

audio = pyaudio.PyAudio()

# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
print("Listening...")


while tray_icon.is_showing:
    data = stream.read(int(CHUNK * 11))
    decoded = np.fromstring(data, 'Float32')

    metadata = load_data.audio_to_metadata(decoded)
    # print(metadata)
    testinput = normalize(np.array([metadata]), axis=1)
    prediction = model.predict(testinput)
    if prediction[0, 1] > 0.8:
        print("Detected a snapped finger ")
        print(prediction)
        # Open facebook
        # os.system("start \"\" https://www.facebook.com")
        width, height = pyautogui.size()
        autopy.mouse.move(width-2, height-2)
        autopy.mouse.click(autopy.mouse.Button.LEFT)
        autopy.mouse.move(width//2, height//2)

print("finished recording")

# # stop Recording
# stream.stop_stream()
# stream.close()
# audio.terminate()
