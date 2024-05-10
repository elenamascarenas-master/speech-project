import os
import librosa
import numpy as np
import shutil
import common_functions as cf


# Path to the folder containing original audio files
folder_path = './release_in_the_wild'

# Path to store the Mel spectrogram numpy files for original audio
nparray_files = './melspec_files'
if not os.path.exists(nparray_files):
    os.mkdir(nparray_files)
elif os.path.exists(nparray_files):
    shutil.rmtree(nparray_files)

# Extract and save Mel spectrogram numpy files for original audio
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    mel_spectrogram = cf.extract_features_melspec(file_path)
    file_name_no_ext = os.path.splitext(file_name)[0]
    new_file_path = os.path.join(nparray_files, file_name_no_ext + '.npy')
    # Save the Mel spectrogram numpy array to the file
    np.save(new_file_path, mel_spectrogram)
# Function to extract Mel spectrogram features

# Path to the folder containing cleaned and padded audio files
folder_path_clean = './release_in_the_wild_clean'

# Path to store the Mel spectrogram numpy files for cleaned and padded audio
nparray_files_clean = './melspec_files_clean'
if not os.path.exists(nparray_files_clean):
    os.mkdir(nparray_files_clean)
elif os.path.exists(nparray_files_clean):
    shutil.rmtree(nparray_files_clean)

# Extract and save Mel spectrogram numpy files for cleaned and padded audio
for file_name in os.listdir(folder_path_clean):
    file_path = os.path.join(folder_path_clean, file_name)
    mel_spectrogram = cf.extract_features_melspec(file_path)
    file_name_no_ext = os.path.splitext(file_name)[0]
    new_file_path = os.path.join(nparray_files_clean, file_name_no_ext + '.npy')
    # Save the Mel spectrogram numpy array to the file
    np.save(new_file_path, mel_spectrogram)


