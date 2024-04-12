import os
import librosa
import numpy as np
import matplotlib.cm as cm

########## creating numpy files to the original files
folder_path = './release_in_the_wild'
nparray_files = './numpy_files'
if not os.path.exists(nparray_files):
    os.mkdir(nparray_files)

durations_origin = []
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    waveform, sample_rate = librosa.load(file_path)
    duration = librosa.get_duration(y=waveform, sr=sample_rate)
    if duration < 1:
        continue
    durations_origin.append(duration)
    mel_spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sample_rate)
    S_dB = librosa.power_to_db(mel_spectrogram, ref=np.min)
    S_normalized = librosa.util.normalize(S_dB)
    # Apply colormap to convert to RGB
    spectrogram_rgb = cm.viridis(S_normalized)[:, :, :3]  # Keep only RGB channels
    # Scale to 0-255 and convert to uint8
    spectrogram_uint8 = (spectrogram_rgb * 255).astype(np.uint8)
    file_name_no_ext = os.path.splitext(file_name)[0]
    file_path = os.path.join(nparray_files, file_name_no_ext + '.npy')
    # Save the numpy array to the file
    np.save(file_path, spectrogram_uint8)
np.save(nparray_files + '_durations.npy', durations_origin)

########## creating numpy files to the cleaned and padded files

folder_path_clean = './release_in_the_wild_clean'
nparray_files_clean = './numpy_files_clean'
if not os.path.exists(nparray_files_clean):
    os.mkdir(nparray_files_clean)

for file_name in os.listdir(folder_path_clean):
    file_path = os.path.join(folder_path_clean, file_name)
    waveform, sample_rate = librosa.load(file_path)
    duration = librosa.get_duration(y=waveform, sr=sample_rate)
    if duration < 1:
        continue
    mel_spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sample_rate)
    S_dB = librosa.power_to_db(mel_spectrogram, ref=np.min)
    S_normalized = librosa.util.normalize(S_dB)
    # Apply colormap to convert to RGB
    spectrogram_rgb = cm.viridis(S_normalized)[:, :, :3]  # Keep only RGB channels
    # Scale to 0-255 and convert to uint8
    spectrogram_uint8 = (spectrogram_rgb * 255).astype(np.uint8)
    file_name_no_ext = os.path.splitext(file_name)[0]
    file_path = os.path.join(nparray_files_clean, file_name_no_ext + '.npy')
    # Save the numpy array to the file
    np.save(file_path, spectrogram_uint8)
