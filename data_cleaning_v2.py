# this code takes about a couple of hours to run,
# it creates cleaned and padded audio files from the original files

import os
import librosa
import numpy as np
import noisereduce as nr
import soundfile as sf

# folder_path is the original audio files,
# clean_folder_path is a new folder that creates for the cleaned and padded audio files
folder_path = './release_in_the_wild'
clean_folder_path = folder_path + '_clean'

# check if needed
DEFAULT_SAMPLE_RATE = 22050

# creating the folder
if not os.path.exists(clean_folder_path):
    os.mkdir(clean_folder_path)


# Function to load audio file and denoise
def load_and_denoise_audio(file_path, target_sr=DEFAULT_SAMPLE_RATE):
    try:
        # Load audio file
        data, rate = librosa.load(file_path, sr=target_sr, mono=True)

        # Check if duration is at least 1 second, if it does, drop.
        duration = librosa.get_duration(y=data, sr=rate)
        if duration < 1:
            print('Warning: audio file too small:', file_path)
            return None

        # Apply noise reduction
        y_denoised = nr.reduce_noise(y=data, sr=rate)

        # will be removed later, not needed
        if len(y_denoised) != len(data):
            raise ValueError("Length of denoised audio does not match length of data")

        return y_denoised
    except Exception as e:
        print(f"Error processing audio file '{file_path}': {e}")
        return None


# Function to pad or truncate audio to target duration
def pad_audio(audio, target_length):
    if len(audio) < target_length:
        # Pad audio if it's shorter than the target duration
        padded_audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
    elif len(audio) > target_length:
        # Truncate audio if it's longer than the target duration
        padded_audio = audio[:target_length]
    else:
        padded_audio = audio
    return padded_audio


# Finding the max length
i = 1
max_length = 0
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    if file_path.endswith('.wav'):
        data, rate = librosa.load(file_path, sr=DEFAULT_SAMPLE_RATE, mono=True)
        if len(data) > max_length:
            max_length = len(data)
    print(max_length)

# creating the cleaned and padded audio and save it to the new folder
for file_name in os.listdir(folder_path):
    file_name_no_ext = os.path.splitext(file_name)[0]
    file_path = os.path.join(folder_path, file_name)
    file_path_clean = os.path.join(clean_folder_path, file_name_no_ext + "_clean.wav")
    print(file_name, "created")
    if file_path.endswith('.wav'):
        audio = load_and_denoise_audio(file_path)
        if audio is None:
            continue
        audio = pad_audio(audio, max_length)
        sf.write(file_path_clean, audio, DEFAULT_SAMPLE_RATE)
