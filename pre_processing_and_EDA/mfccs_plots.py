import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
from matplotlib import cm

def truncate_mel_spectrogram(mfccs, num_timeframes):
    truncated_mfccs = mfccs[:, :num_timeframes]
    return truncated_mfccs

def plot_mfccs(mfccs_raw, person, bona_fide=True, truncated=True, num_timeframes=300):
    if truncated:
        mfccs = truncate_mel_spectrogram(mfccs_raw,num_timeframes)
    else:
        mfccs = mfccs_raw

    audio_type = 'Real' if bona_fide else 'Fake'
    truncated_string = 'Truncated' if truncated else ''

    plt.figure(figsize=(10, 6))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title(person + ' MFCCs - ' + audio_type + ' ' + truncated_string)
    plt.xlabel('Time')
    plt.ylabel('MFCC Coefficients')
    plt.tight_layout()
    plt.show()

def mfccs_aggregation(folder_path, filter_string):
    # Initialize an empty list to store Mel spectrogram arrays
    mfccs_arrays = []

    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        if filter_string in file_name and file_name.endswith('.npy'):
            # Load the Mel spectrogram from the file
            mfccs = np.load(os.path.join(folder_path, file_name))
            # Append the Mel spectrogram array to the list
            mfccs_arrays.append(mfccs)

    # Aggregate the Mel spectrogram arrays into a single array
    #norm_aggregation_mean = np.mean(mel_spectrogram_arrays, axis=0).astype(np.float32) / 255.0
    mean_mfccs = np.mean(mfccs_arrays, axis=0)

    # Return the aggregated Mel spectrogram array and the plot object
    return mean_mfccs

# Example usage:
folder_path = 'mfccs_files_clean'
bona_fide_agg = mfccs_aggregation(folder_path, "bona-fide")
spoof_agg = mfccs_aggregation(folder_path, "spoof")


plot_mfccs(bona_fide_agg, 'Aggregated', bona_fide=True, truncated=True,num_timeframes=300)
plot_mfccs(spoof_agg, 'Aggregated', bona_fide=False, truncated=True,num_timeframes=300)


#Person 1: 2pac
person1 = '2pac'
filter_bona_fide = person1+ '_bona-fide'
filter_spoof = person1+ '_spoof'
bona_fide_person1 = mfccs_aggregation(folder_path, filter_bona_fide)
spoof_person1 = mfccs_aggregation(folder_path, filter_spoof )
plot_mfccs(bona_fide_person1, person1, bona_fide=True, truncated=True,num_timeframes=300)
plot_mfccs(spoof_person1, person1, bona_fide=False, truncated=True,num_timeframes=300)


#Person 2: Clinton
person2 = 'bill_clinton'
filter_bona_fide = person2+ '_bona-fide'
filter_spoof = person2+ '_spoof'
bona_fide_person2 = mfccs_aggregation(folder_path, filter_bona_fide)
spoof_person2 = mfccs_aggregation(folder_path, filter_spoof )
plot_mfccs(bona_fide_person2, person2, bona_fide=True, truncated=True,num_timeframes=300)
plot_mfccs(spoof_person2, person2, bona_fide=False, truncated=True,num_timeframes=300)


#Person 3: Billie Eilish
person3 = 'billie_eilish'
filter_bona_fide = person3+'_bona-fide'
filter_spoof = person3+'_spoof'
bona_fide_person3 = mfccs_aggregation(folder_path, filter_bona_fide)
spoof_person3 = mfccs_aggregation(folder_path, filter_spoof )
plot_mfccs(bona_fide_person3, person3, bona_fide=True, truncated=True,num_timeframes=300)
plot_mfccs(spoof_person3, person3, bona_fide=False, truncated=True,num_timeframes=300)

