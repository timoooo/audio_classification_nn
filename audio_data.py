import os
import shutil

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


def convertPNGtoArray(img):
    return np.ndarray.flatten(np.asarray(img))


def getFeatureSpectrogram(x, genre, number):
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
    plt.close()
    return features


def getFeaturesMean(x, sr):
    # SpectralCentroid, SpectralRolloff, SpectralBandwidth, ZeroCrossingRate, ChromaFeatures, MFFC{20}, Tempo, beats
    features = []
    features.append(np.mean(get_spectral_centroid(x, sr=sr)[0]))
    features.append(np.mean(get_spectral_rolloff(x, sr=sr)[0]))
    features.append(np.mean(get_spectral_bandwidth(x, sr=sr)[0]))
    features.append(np.mean(librosa.feature.chroma_stft(x, sr=sr)))
    features.append(get_mean_zero_crossing(x))

    mffcs = librosa.feature.mfcc(x, sr=sr)
    for array in mffcs:
        features.append(np.mean(array))

    # tempo and beat extraction
    x_harmonic, x_percussive = librosa.effects.hpss(x)
    tempo, beats = librosa.beat.beat_track(y=x_percussive, sr=sr)

    features.append(np.mean(tempo))
    features.append(np.mean(beats))

    return features


def get_mean_zero_crossing(x):
    zero_crossing = librosa.zero_crossings(x, pad=False)
    converted_array = [int(elem) for elem in zero_crossing]  # convert T/F to 1/0
    bumped_array = [elem * 10000 for elem in converted_array]
    mean_zero_crossing = np.mean(bumped_array)
    return mean_zero_crossing


def get_spectral_centroid(x, sr):
    return librosa.feature.spectral_centroid(x, sr=sr)[0][:1290]


def get_spectral_rolloff(x, sr):
    return librosa.feature.spectral_rolloff(x, sr=sr)[0][:1290]


def get_spectral_bandwidth(x, sr):
    return librosa.feature.spectral_bandwidth(x, sr=sr)[0][:1290]


def get_zero_crossing_rate(x, sr):
    zerocrossing_temp = librosa.zero_crossings(x, pad=False)[:660000]
    # goal about 30k elements
    converted_array = [int(elem) for elem in zerocrossing_temp]  # convert T/F to 1/0
    bumped_array = [elem * 10000 for elem in converted_array]
    flatten_zerocrossing = [np.mean(bumped_array[x:x + 5]) for x in range(0, len(bumped_array), 20)]
    return flatten_zerocrossing


def get_MFCCs(x, sr):
    mffcs_temp = librosa.feature.mfcc(x, sr=sr)
    flatten_mffcs = [mffc_array for sublist in mffcs_temp for mffc_array in sublist]
    return flatten_mffcs[:25800]


def get_chroma_features(x, sr):
    return librosa.feature.chroma_stft(x, sr=sr)[:1290]


# Features
def getFeatures(x, sr):
    feature_dict = {
        "spectralcentroid": get_spectral_centroid(x, sr),
        "spectralrolloff": get_spectral_rolloff(x, sr),
        "spectralbandwidth": get_spectral_bandwidth(x, sr),
        "zerocrossingrate": get_zero_crossing_rate(x, sr),
        "mfcc": get_MFCCs(x, sr),
        "chroma": get_chroma_features(x, sr)
    }

    return feature_dict


def save_to_csv(data):
    panda_df = pd.DataFrame(data)
    panda_df.to_csv(path_or_buf=".\audio_data_means.csv")


# Creates Huge amount of data if saved to csv
def getAudioData():
    # Adapt to path
    rootdir = "genres"
    return_array = []
    for subdir, dirs, files in os.walk(rootdir):
        for i in tqdm(range(files.__len__())):
            genre = files[i].split(".")[0]
            audio_file = os.path.join("genres", genre, files[i])
            audio_data, sr = librosa.load(audio_file, sr=None, dtype=np.float64)
            audio_tuple = (getFeatures(audio_data, sr), genre)
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
            getFeatureSpectrogram(audio_data, genre, i)
    return print("Done")


generateImages()
#result = getAudioDataMeans()
#save_to_csv(result)
