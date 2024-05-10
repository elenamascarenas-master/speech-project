import os
import librosa
import numpy as np
import shutil
import matplotlib.pyplot as plot

def extract_features_melspec(audio_path):
    waveform, sample_rate = librosa.load(audio_path, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=40)
    mel_spectrogram = librosa.amplitude_to_db(mel_spectrogram, ref=np.min)

    return mel_spectrogram

def extract_features_mfcc(audio_path):
    waveform, sample_rate = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=13)
    duration = librosa.get_duration(y= waveform, sr=sample_rate)
    return mfccs, duration


def truncate_mel_spectrogram(mel_spectrogram, num_timeframes):
    truncated_mel_spectrogram = mel_spectrogram[:, :num_timeframes]
    return truncated_mel_spectrogram


def extract_duration(audio_path):
    waveform, sample_rate = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=waveform, sr=sample_rate)
    return duration
