import os
import librosa
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import common_functions as cf
import matplotlib.pyplot as plt
import shutil



# Define folder paths for human and spoof audio files
folder_path = '../release_in_the_wild'
human_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if "bona-fide" in file]
spoof_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if "spoof" in file]

# Extract features for all files
all_features = []
labels = []
for file in human_files:
    features = extract_features_melspec(file)
    all_features.append(features)
    labels.append('real')
for file in spoof_files:
    features = extract_features_melspec(file)
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
folder_path_clean = '../release_in_the_wild_clean'
human_files_clean = [os.path.join(folder_path_clean, file) for file in os.listdir(folder_path_clean) if "bona-fide" in file]
spoof_files_clean = [os.path.join(folder_path_clean, file) for file in os.listdir(folder_path_clean) if "spoof" in file]

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

#Histogram
duration_bona_fide = []

for filename in os.listdir(folder_path):
    if 'bona-fide' in filename:
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            waveform, sample_rate = librosa.load(filepath, sr=None)
            duration = librosa.get_duration(y= waveform, sr=sample_rate)
            duration_bona_fide.append(duration)

duration_spoof = []

for filename in os.listdir(folder_path):
    if 'spoof' in filename:
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            waveform, sample_rate = librosa.load(filepath, sr=None)
            duration = librosa.get_duration(y= waveform, sr=sample_rate)
            duration_spoof.append(duration)


plt.hist(duration_bona_fide, bins=30, alpha=0.5, color='#0066b2',label='Real Audios')
plt.hist(duration_spoof, bins=30, alpha=0.5, color='#E32636', label='Fake Audios')
plt.title("Real vs. Fake Audio duration histogram")
plt.xlabel('Duration in seconds')
plt.xticks(range(0,26))
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.show()
# Show plot
plt.show()

#Counting files
counter_bona_fide=0
for filename in os.listdir(folder_path):
    if 'bona-fide' in filename:
        counter_bona_fide=counter_bona_fide+1

print(counter_bona_fide)

counter_spoof=0
for filename in os.listdir(folder_path):
    if 'spoof' in filename:
        counter_spoof=counter_spoof+1

print(counter_spoof)

#mel spectrogram example after cleaning
noised = '../release_in_the_wild/billie_eilish_bona-fide_729.wav'
duration_raw = cf.extract_duration(noised)
denoised = '../release_in_the_wild_clean/billie_eilish_bona-fide_729_clean.wav'
melspec_raw = cf.extract_features_melspec(noised)

seconds = duration_raw
sr=22050
hop_length=512
time_frames = int(seconds * sr / hop_length)

melspec_clean = cf.truncate_mel_spectrogram(cf.extract_features_melspec(denoised),time_frames)


plt.figure(figsize=(10, 4))
librosa.display.specshow(melspec_raw, x_axis='time', y_axis='mel', cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.title('Billie Eilish Raw' + ' Mel Spectrogram ')
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Hz)')
plt.show()

plt.figure(figsize=(10, 4))
librosa.display.specshow(melspec_clean, x_axis='time', y_axis='mel', cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.title('Billie Eilish Clean' + ' Mel Spectrogram ')
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Hz)')
plt.show()