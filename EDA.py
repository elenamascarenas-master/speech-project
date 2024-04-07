# After cleaning
# This is the file for EDA and visualization of the data

import os
import librosa
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Roxane code for visualization PC1 vs PC2
def extract_features(audio_path):
    # Extrait les MFCC (ou tout autre descripteur audio)
    waveform, sample_rate = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=13)
    # Retourne les MFCC moyens pour réduire la dimension
    return np.mean(mfccs, axis=1)

# Dossier contenant les fichiers audio des humains
human_dir = "C:/Users/dawdi/Desktop/Artist/human"
human_files = [os.path.join(human_dir, file) for file in os.listdir(human_dir)]

# Dossier contenant les fichiers audio des faux
spoof_dir = "C:/Users/dawdi/Desktop/Artist/spoof"
spoof_files = [os.path.join(spoof_dir, file) for file in os.listdir(spoof_dir)]

# Extraction des caractéristiques pour tous les fichiers
all_features = []
labels = []
for file in human_files:
    features = extract_features(file)
    all_features.append(features)
    labels.append('human')
for file in spoof_files:
    features = extract_features(file)
    all_features.append(features)
    labels.append('spoof')

# Réduction de la dimensionnalité avec PCA
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(all_features)

# Plot des données réduites
plt.figure(figsize=(10, 6))
for label in np.unique(labels):
    idx = np.where(np.array(labels) == label)
    plt.scatter(reduced_features[idx, 0], reduced_features[idx, 1], label=label)
plt.title('Visualisation des Caractéristiques Audio')
plt.xlabel('Composante Principale 1')
plt.ylabel('Composante Principale 2')
plt.legend()
plt.show()
