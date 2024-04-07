import os
import pandas as pd
import librosa
import numpy as np
import noisereduce as nr
import time
import soundfile as sf

folder_path = './release_in_the_wild'
#output_folder = './denoised_audio'

# Function to load audio file and denoise
def load_and_denoise_audio(file_path, target_sr=22050):
    try:
        # Load audio file
        data, rate = librosa.load(file_path, sr=target_sr, mono=True)

        # Check if duration is at least 1 second
        duration = librosa.get_duration(y=data, sr=rate)
        if duration < 1:
            return None

        # Apply noise reduction
        y_denoised = nr.reduce_noise(y=data, sr=rate)

        #output_path = os.path.join(output_folder, f'{file_name[:-4]}_reduced_noise.wav')
        #sf.write(output_path, y_denoised, rate)

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


# Process folder with audio files
start_time = time.time()

audio_data = []

for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    if file_path.endswith('.wav'):
        audio = load_and_denoise_audio(file_path)
        if audio is not None:
            audio_data.append(audio)

# Calculate maximum length of audio objects
max_length = max(len(audio) for audio in audio_data)

# Pad audio objects to have the same dimensions
padded_audio_data = [pad_audio(audio, max_length) for audio in audio_data]

# Create DataFrame
df = pd.DataFrame({'Audio': padded_audio_data})
print(df)

# Optional: Save DataFrame to a CSV file
# df.to_csv('clean_and_padded_audio.csv', index=False)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time} seconds")
