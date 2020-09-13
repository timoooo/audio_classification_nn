import os
import shutil
import warnings
from asyncio import wait
from concurrent.futures.thread import ThreadPoolExecutor
import pathlib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import re
import pandas as pd

# 1 0 0 0 0 0 0 0 0 0    blues
# 0 1 0 0 0 0 0 0 0 0    classical
# 0 0 1 0 0 0 0 0 0 0    country
# 0 0 0 1 0 0 0 0 0 0    disco
# 0 0 0 0 1 0 0 0 0 0    hiphop
# 0 0 0 0 0 1 0 0 0 0    jazz
# 0 0 0 0 0 0 1 0 0 0    metal
# 0 0 0 0 0 0 0 1 0 0    pop
# 0 0 0 0 0 0 0 0 1 0    reggae
# 0 0 0 0 0 0 0 0 0 1    rock
warnings.filterwarnings("ignore")


def getGenre(str):
    switcher = {
        "blues": np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).transpose(),
        "classical": np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]).transpose(),
        "country": np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]).transpose(),
        "disco": np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]).transpose(),
        "hiphop": np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]).transpose(),
        "jazz": np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]).transpose(),
        "metal": np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]).transpose(),
        "pop": np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]).transpose(),
        "reggae": np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]).transpose(),
        "rock": np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]).transpose(),
    }

    return switcher.get(str)


def convertPNGtoArray(img):
    return np.ndarray.flatten(np.asarray(img))


def getFeaturesPlots(x, sr, genre, number):
    features = []
    image_path = "images/" + genre + "-" + str(number)
    # spectrogram
    plt.figure(figsize=(3.2, 2.4))
    cmap = plt.get_cmap('inferno')
    plt.specgram(x, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB')
    plt.savefig(image_path + "-spectrogram.png")
    plt.clf()
    image = Image.open(image_path + "-spectrogram.png")
    features.append(convertPNGtoArray(image))
    # mel-scaled spectrogram
    plt.figure(figsize=(3.2, 2.4))
    mfcc = librosa.feature.mfcc(x, sr)
    librosa.display.specshow(mfcc, x_axis='time', y_axis='hz')
    plt.savefig(image_path + "-mfcc.png")
    plt.clf()
    image = Image.open(image_path + "-mfcc.png")
    features.append(convertPNGtoArray(image))
    # Chromagram
    plt.figure(figsize=(3.2, 2.4))
    chromagram = librosa.feature.chroma_stft(x, sr=sr)
    librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma')
    plt.savefig(image_path + "-chroma.png")
    image = Image.open(image_path + "-chroma.png")
    features.append(convertPNGtoArray(image))
    plt.clf()
    return features


def getFeatureSpectrogram(x, sr, genre, number):
    features = []
    image_name = str(number)
    # spectrogram
    plt.figure(figsize=(6.4, 4.8))
    cmap = plt.get_cmap('inferno')
    plt.specgram(x, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB')
    plt.axis('off');
    if not os.path.exists("images/" + genre):
        os.mkdir("images/" + genre)
    plt.savefig("images/" + genre + "/" + image_name + ".png", bbox_inches='tight', pad_inches=0)
    return features


def getFeaturesMean(x, sr):
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(x, sr=sr)[0])
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(x, sr=sr)[0])
    spectral_bandwidth = np.mean(librosa.feature.spectral_rolloff(x, sr=sr)[0])
    zero_crossing = librosa.zero_crossings(x, pad=False)
    converted_array = [int(elem) for elem in zero_crossing]  # convert T/F to 1/0
    bumped_array = [elem * 10000 for elem in converted_array]
    #bumped zero crossing values might be misleading data
    mean_zero_crossing = np.mean(bumped_array)
    chroma_features = np.mean(librosa.feature.chroma_stft(x, sr=sr))
    features = [spectral_centroid, spectral_rolloff, spectral_bandwidth, mean_zero_crossing, chroma_features]
    mffcs = librosa.feature.mfcc(x, sr=sr)
    for array in mffcs:
        features.append(np.mean(array))
    onset_env = librosa.onset.onset_strength(x, sr)
    # tempo and beat extraction
    x_harmonic, x_percussive = librosa.effects.hpss(x)
    tempo, beats = librosa.beat.beat_track(y=x_percussive, sr=sr)
    features.append(np.mean(tempo))
    features.append(np.mean(beats))
    return features


def getSpectralCentroid(x, sr):
    return librosa.feature.spectral_centroid(x, sr=sr)[0][:1290]


def getSpectralRolloff(x, sr):
    return librosa.feature.spectral_rolloff(x, sr=sr)[0][:1290]


def getSpectralBandwidth(x, sr):
    return librosa.feature.spectral_bandwidth(x, sr=sr)[0][:1290]


def getZeroCrossingRate(x, sr):
    zerocrossing_temp = librosa.zero_crossings(x, pad=False)[:660000]
    # goal about 30k elements
    converted_array = [int(elem) for elem in zerocrossing_temp]  # convert T/F to 1/0
    bumped_array = [elem * 10000 for elem in converted_array]
    flatten_zerocrossing = [np.mean(bumped_array[x:x + 5]) for x in range(0, len(bumped_array), 20)]
    return flatten_zerocrossing


def getMFCCs(x, sr):
    mffcs_temp = librosa.feature.mfcc(x, sr=sr)
    flatten_mffcs = [mffc_array for sublist in mffcs_temp for mffc_array in sublist]
    return flatten_mffcs[:25800]


def getChromaFeatures(x, sr):
    return librosa.feature.chroma_stft(x, sr=sr)[:1290]


# Features
def getFeatures(x, sr):
    feature_dict = {
        "spectralcentroid": getSpectralCentroid(x, sr),
        "spectralrolloff": getSpectralRolloff(x, sr),
        "spectralbandwidth": getSpectralBandwidth(x, sr),
        "zerocrossingrate": getZeroCrossingRate(x, sr),
        "mfcc": getMFCCs(x, sr),
        "chroma": getChromaFeatures(x, sr)
    }

    return feature_dict


def save_to_csv(data):
    panda_df = pd.DataFrame(data)
    panda_df.to_csv(path_or_buf=".\audio_data_means.csv")


def getAudioData():
    # Adapt to path
    rootdir = "genres"
    return_array = []
    genre_list = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    for subdir, dirs, files in os.walk(rootdir):
        if (len(files) > 0):
            genre = subdir.split("\\", 2)[1]
            print('(' + str(genre_list.index(genre) + 1) + '/10):', genre)
            for i in tqdm(range(files.__len__())):
                audio_file = os.path.join(subdir, files[i])
                audio_data, sr = librosa.load(audio_file, sr=None, dtype=np.float64)
                # audio_data = np.array([audio_data]).transpose()  # pseudo matrix
                audio_genre = getGenre(genre)
                audio_tuple = (getFeatures(audio_data, sr), audio_genre)
                return_array.append(audio_tuple)
    return return_array


def getAudioDataMeans():
    # Adapt to path
    rootdir = "genres"
    result = []
    for subdir, dirs, files in os.walk(rootdir):
        for i in tqdm(range(files.__len__())):
            genre = files[i].split(".")[0]
            audio_file = os.path.join("genres", genre, files[i])
            audio_data, sr = librosa.load(audio_file, sr=None, dtype=np.float64)
            features = getFeaturesMean(audio_data, sr)
            audio_tuple = (features, genre)
            result.append(audio_tuple)
    return result


def generateImages():
    # WARNING  THIS WILL DELETE THE IMAGES FOLDER AND RECREATE THE PLOTS
    if (os.path.exists("images/")):
        print("Deleting old images folder")
        shutil.rmtree('images/', ignore_errors=True)
        print("Recreating images folder")
        os.mkdir("images")
    else:
        print("Creating images folder")
        os.mkdir("images")

    # Adapt to path
    for root, dirs, files in os.walk("genres"):
        for i in tqdm(range(files.__len__())):
            genre = files[i].split(".")[0]

            audio_file = os.path.join("genres", genre, files[i])
            audio_data, sr = librosa.load(audio_file, sr=None, dtype=np.float64)
            getFeatureSpectrogram(audio_data, sr, genre, i)
    return print("Done")



generateImages()
result = getAudioDataMeans()
save_to_csv(result)
