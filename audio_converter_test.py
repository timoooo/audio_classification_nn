import os
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from matplotlib import cm,axis as ax
import os
from asyncio import wait
from concurrent.futures.thread import ThreadPoolExecutor
import librosa.display
import librosa
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
audio_data = 'genres/blues/blues.00001.wav'
x, sr = librosa.load(audio_data, sr=None)

#Spectogram
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()
plt.clf()


#Plot the signal:
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)
# Zooming in
n0 = 9000
n1 = 9100
plt.figure(figsize=(14, 5))
plt.plot(x[n0:n1])
plt.grid()

chromagram = librosa.feature.chroma_stft(x, sr=sr)
plt.figure(figsize=(15, 5))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma',hop_length=512, cmap='coolwarm')
plt.show()


# default sample rate = 22050
# print(type(x), str(sr))


# visualize audio file
# plt.figure(figsize=(14, 5))
# librosa.display.waveplot(x, sr=sr)

# plt.show()

# spectrogram = loudness over time

# X = librosa.stft(x)
# Xdb = librosa.amplitude_to_db(abs(X))
# plt.figure(figsize=(14, 5))
# librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
# plt.colorbar()
features = []














# plt.show()
#plt.specgram(x, NFFT=2048, Fs=2, Fc=0, noverlap=128, sides='default', mode='default', scale='dB');
#plt.show()
print("Chromagram length:" + str(len(chromagram[1])))

rootdir = "genres"
for subdir, dirs, files in os.walk(rootdir):
    if (len(files) > 0):
        for file in files:
            audio_file = os.path.join(subdir, file)
            audio_data, sr = librosa.load(audio_file, sr=None, dtype=np.float64)
            if (sr != 22050):
                print(str(sr))
        print("Genre Done")

##Feature extraction
# spectral centroid

# .spectral_centroid will return an array with columns equal to a number of frames present in your sample.
spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)
print("Spectral centroids length:" + str(len(spectral_centroids)))  # THE MINUM IS  1290
# Spectral Rolloff
# It is a measure of the shape of the signal. It represents the frequency at which high frequencies decline to 0.
# To obtain it, we have to calculate the fraction of bins in the power spectrum where 85% of its power is at lower frequencies.

spectral_rolloff = librosa.feature.spectral_rolloff(x + 0.01, sr=sr)[0]
print("Spectral rolloff length:" + str(len(spectral_rolloff)))  # THE MINIMUM IS 1290

# Spectral Bandwidth
# https://miro.medium.com/max/515/1*oUtYY0-j6iEc78Dew3d0uA.png

spectral_bandwidth = librosa.feature.spectral_bandwidth(x + 0.01, sr=sr)[0]
print("Spectral bandwidth length:" + str(len(spectral_bandwidth)))  # THE MINIMUM IS 1290
# Zero-Crossing Rate

zero_crossings = librosa.zero_crossings(x, pad=False)  # THE MINIMUM IS 660 000
# zero_crossing_byte_array = bytearray(struct.pack("f", zero_crossings))
print("ZeroCrossing length:" + str(zero_crossings))

# Mel-Frequency Cepstral Coefficients
mfccs = librosa.feature.mfcc(x, sr=sr)  # The minimum mfccs list size is 20   the min sublist size is 1290
print("MFFCs length:" + str(len(mfccs)))

# Chroma Frequencies
chromagram = librosa.feature.chroma_stft(x, sr=sr)  #
print("Chromagram length:" + str(len(chromagram)))  # THE MINIMUM IS 1290

mffcs_temp = librosa.feature.mfcc(x, sr=sr)
flatten_mffcs = [item for sublist in mffcs_temp for item in sublist]
len(flatten_mffcs)

zerocrossing_temp = librosa.zero_crossings(x, pad=False)
# goal about 30k elements
flatten_zerocrossing = []
converted_array = [int(elem) for elem in zerocrossing_temp]  # convert T/F to 1/0
min_index = 0

sublist = [np.mean(converted_array[x:x + 5]) for x in range(0, len(converted_array), 20)]
len(sublist)
for i in range(0, len(converted_array), 20):
    flatten_zerocrossing.append(np.mean(converted_array[min_index:i]))
    min_index = i + 1
