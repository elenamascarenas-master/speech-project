import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import pandas as pd
import common_functions as cf

def plot_melspectrogram(mel_spectrogram_raw, person, bona_fide=True, truncated=True, num_timeframes=300):
    if truncated:
        mel_spectrogram = cf.truncate_mel_spectrogram(mel_spectrogram_raw,num_timeframes)
    else:
        mel_spectrogram = mel_spectrogram_raw

    audio_type = 'Real' if bona_fide else 'Fake'
    truncated_string = 'Truncated' if truncated else ''

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram, x_axis='time', y_axis='mel', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title(person + ' Mel Spectrogram - ' + audio_type + ' ' + truncated_string)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    plt.show()

def melspec_aggregation(folder_path,names_array,audio_type):
    # Initialize an empty list to store Mel spectrogram arrays
    mel_spectrogram_arrays = []

    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        if any(name_string in file_name for name_string in names_array) and file_name.endswith('.npy') and audio_type in file_name:
            # Load the Mel spectrogram from the file
            mel_spectrogram = np.load(os.path.join(folder_path, file_name))
            # Append the Mel spectrogram array to the list
            mel_spectrogram_arrays.append(mel_spectrogram)

    # Aggregate the Mel spectrogram arrays into a single array
    #norm_aggregation_mean = np.mean(mel_spectrogram_arrays, axis=0).astype(np.float32) / 255.0
    norm_aggregation_mean = np.mean(mel_spectrogram_arrays, axis=0)

    # Return the aggregated Mel spectrogram array and the plot object
    return norm_aggregation_mean

#General parameters
folder_path = './melspec_files_clean'
seconds = 5
sr=22050
hop_length=512
time_frames = int(seconds * sr / hop_length)

#Aggregated Mel Spec
bona_fide_agg = melspec_aggregation(folder_path, "bona-fide")
spoof_agg = melspec_aggregation(folder_path, "spoof")
plot_melspectrogram(bona_fide_agg, 'Aggregated', bona_fide=True, truncated=True,num_timeframes=time_frames)
plot_melspectrogram(spoof_agg, 'Aggregated', bona_fide=False, truncated=True,num_timeframes=time_frames)


#Mel Spectrograms specific celebrities

#Person 1: 2pac
person1 = '2pac'
bona_fide_person1 = melspec_aggregation(folder_path, list(person1), 'bona-fide')
spoof_person1 = melspec_aggregation(folder_path, list(person1), 'spoof')
plot_melspectrogram(bona_fide_person1, str(person1), bona_fide=True, truncated=True,num_timeframes=time_frames)
plot_melspectrogram(spoof_person1, person1, bona_fide=False, truncated=True,num_timeframes=time_frames)

#Person 2: Clinton
person2 = ['bill_clinton']
bona_fide_person2 = melspec_aggregation(folder_path, list(person2), 'bona-fide')
spoof_person2 = melspec_aggregation(folder_path, list(person2), 'spoof')
plot_melspectrogram(bona_fide_person2, person2, bona_fide=True, truncated=True,num_timeframes=time_frames)
plot_melspectrogram(spoof_person2, person2, bona_fide=False, truncated=True,num_timeframes=time_frames)


#Person 3: Billie Eilish
person3 = ['billie_eilish']
bona_fide_person3 = melspec_aggregation(folder_path, list(person3), 'bona-fide')
spoof_person3 = melspec_aggregation(folder_path, list(person3), 'spoof')
plot_melspectrogram(bona_fide_person3, person3, bona_fide=True, truncated=True,num_timeframes=time_frames)
plot_melspectrogram(spoof_person3, person3, bona_fide=False, truncated=True,num_timeframes=time_frames)


celeb_df = pd.read_csv('celebrity_data.csv')


