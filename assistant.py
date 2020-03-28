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
# import requests
import json
# from gtts import gTTS
import os
import speech_recognition as sr 
sample_rate = 48000
chunk_size = 2048


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

with sr.Microphone(device_index = device_id, sample_rate = sample_rate,  
                            chunk_size = chunk_size) as source: 
        print ("Bạn:")
        #wait for a second to let the recognizer adjust the  
        #energy threshold based on the surrounding noise level 
        r.adjust_for_ambient_noise(source) 
        #listens for the user's input 
        audio = r.listen(source) 
        try: 
            text = r.recognize_google(audio, language='vi-VN') 
            print ('  +--->'+text)
            reply = bot_rep(text)
            print_and_speak(reply)
        #error occurs when google could not understand what was said 

        except sr.UnknownValueError: 
            print('  +--->%&UH&7GBY&76%*y%^&H**YUJU*U&G')
            print_and_speak("Xin lỗi tôi không hiểu bạn nói gì", file='sorry.mp3') 
#             print("")
        except sr.RequestError as e: 
            print("Không thể kết nối Google; {0}".format(e))

# first_snap = False
# last_snap = 0
# while tray_icon.is_showing:
    # data = stream.read(int(CHUNK * 11))
    # decoded = np.fromstring(data, 'Float32')

    # metadata = load_data.audio_to_metadata(decoded)
    ## print(metadata)
    # testinput = normalize(np.array([metadata]), axis=1)
    # prediction = model.predict(testinput)

    # if prediction[0, 1] > 0.8:
        # if time.time() - last_snap > 2.3 : first_snap = False
        # last_snap = time.time()
        # print("Detected strong snapped finger ")
        # print(prediction)
        # # Open facebook
        # os.system("start \"\" https://www.youtube.com/?gl=VN")
        # first_snap = not first_snap
        # if not first_snap: on_snapped_finger()
    # if prediction[0, 1] > 0.5:
        # print("Detected a snapped finger ")
        # print(prediction)

# print("finished recording")

# # stop Recording
# stream.stop_stream()
# stream.close()
# audio.terminate()
