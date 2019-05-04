from pydub import AudioSegment
oriAudio = AudioSegment.from_wav("./data/iphone.wav")
for time_start in range(0, len(oriAudio)-500, 500):
    newAudio = oriAudio[time_start:time_start+500]
    newAudio.export('./data/raw/{}.wav'.format("iphone_" + str(time_start//500)), format="wav") #Exports to a wav file in the current path.
    print("Exported file {}".format("iphone_" + str(time_start//500) + ".wav"))