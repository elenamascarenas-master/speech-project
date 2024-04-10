import os
import librosa
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Function to extract features from audio files
def extract_features(audio_path):
    waveform, sample_rate = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=13)
    return np.mean(mfccs, axis=1)

# Define folder paths for human and spoof audio files
folder_path = './release_in_the_wild'
human_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if "bona-fide" in file]

spoof_dir = "./release_in_the_wild"
spoof_files = [os.path.join(spoof_dir, file) for file in os.listdir(spoof_dir) if "spoof" in file]

# Extract features for all files
all_features = []
labels = []
for file in human_files:
    features = extract_features(file)
    all_features.append(features)
    labels.append('real')
for file in spoof_files:
    features = extract_features(file)
    all_features.append(features)
    labels.append('spoof')

all_features= pd.DataFrame(all_features)

# Reduce dimensionality with PCA
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(all_features)

# Plot the reduced features
plt.figure(figsize=(10, 6))
for label in np.unique(labels):
    idx = np.where(np.array(labels) == label)
    if label == 'real':
        color = 'blue'
    else:
        color = 'red'
    plt.scatter(reduced_features[idx, 0], reduced_features[idx, 1], label=label, color=color)
plt.title('Visualization of Audio Features Before Cleaning')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# After cleaning visualization
folder_path_clean = './release_in_the_wild_clean'
human_files_clean = [os.path.join(folder_path_clean, file) for file in os.listdir(folder_path_clean) if "bona-fide" in file]

spoof_dir_clean = "./release_in_the_wild_clean"
spoof_files_clean = [os.path.join(spoof_dir_clean, file) for file in os.listdir(spoof_dir_clean) if "spoof" in file]

# Extract features for all files
all_features_clean = []
labels_clean = []
for file in human_files_clean:
    features = extract_features(file)
    all_features_clean.append(features)
    labels_clean.append('real')
for file in spoof_files_clean:
    features = extract_features(file)
    all_features_clean.append(features)
    labels_clean.append('spoof')

all_features_clean= pd.DataFrame(all_features)

# Reduce dimensionality with PCA
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(all_features_clean)

# Plot the reduced features
plt.figure(figsize=(10, 6))
for label in np.unique(labels_clean):
    idx = np.where(np.array(labels_clean) == label)
    if label == 'real':
        color = 'blue'
    else:
        color = 'red'
    plt.scatter(reduced_features[idx, 0], reduced_features[idx, 1], label=label, color=color)
plt.title('Visualization of Audio Features After Cleaning')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()