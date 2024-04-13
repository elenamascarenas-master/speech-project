import os
import numpy as np
import matplotlib.pyplot as plt

def truncate_mel_spectrogram(mel_spectrogram, num_timeframes):
    truncated_mel_spectrogram = mel_spectrogram[:, :num_timeframes]
    return truncated_mel_spectrogram

def plot_melspectrogram(mel_spectrogram_raw, person, bona_fide=True, truncated=True, num_timeframes=300):
    if truncated:
        mel_spectrogram = truncate_mel_spectrogram(mel_spectrogram_raw,num_timeframes)
    else:
        mel_spectrogram = mel_spectrogram_raw

    audio_type = 'Real' if bona_fide else 'Fake'
    truncated_string = 'Truncated' if truncated else ''

    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spectrogram, aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title(person + ' Mel Spectrogram - ' + audio_type + ' ' + truncated_string)
    plt.xlabel('Time (frame)')
    plt.ylabel('Mel Frequency Bin')
    plt.show()

def plot_aggregated_mel_spectrogram(folder_path, filter_string):
    # Initialize an empty list to store Mel spectrogram arrays
    mel_spectrogram_arrays = []

    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        if filter_string in file_name and file_name.endswith('.npy'):
            # Load the Mel spectrogram from the file
            mel_spectrogram = np.load(os.path.join(folder_path, file_name))
            # Append the Mel spectrogram array to the list
            mel_spectrogram_arrays.append(mel_spectrogram)

    # Aggregate the Mel spectrogram arrays into a single array
    norm_aggregation_mean = np.mean(mel_spectrogram_arrays, axis=0).astype(np.float32) / 255.0

    # Return the aggregated Mel spectrogram array and the plot object
    return norm_aggregation_mean
# Example usage:
folder_path = './numpy_files_clean'
bona_fide_agg = plot_aggregated_mel_spectrogram(folder_path, "bona-fide")
spoof_agg = plot_aggregated_mel_spectrogram(folder_path, "spoof")

plot_melspectrogram(bona_fide_agg, 'Aggregated', bona_fide=True, truncated=True,num_timeframes=300)
plot_melspectrogram(spoof_agg, 'Aggregated', bona_fide=False, truncated=True,num_timeframes=300)


#Mel Spectograms specific celebrities

#Person 1: 2pac
person1 = '2pac'
filter_bona_fide = person1+ '_bona-fide'
filter_spoof = person1+ '_spoof'
bona_fide_person1 = plot_aggregated_mel_spectrogram(folder_path, filter_bona_fide)
spoof_person1 = plot_aggregated_mel_spectrogram(folder_path, filter_spoof )
plot_melspectrogram(bona_fide_person1, person1, bona_fide=True, truncated=True,num_timeframes=300)
plot_melspectrogram(spoof_person1, person1, bona_fide=False, truncated=True,num_timeframes=300)

truncate_mel_spectrogram(bona_fide_agg,300)
#Person 2: Clinton
person2 = 'bill_clinton'
filter_bona_fide = person2+ '_bona-fide'
filter_spoof = person2+ '_spoof'
bona_fide_person2 = plot_aggregated_mel_spectrogram(folder_path, filter_bona_fide)
spoof_person2 = plot_aggregated_mel_spectrogram(folder_path, filter_spoof )
plot_melspectrogram(bona_fide_person2, person2, bona_fide=True, truncated=True,num_timeframes=300)
plot_melspectrogram(spoof_person2, person2, bona_fide=False, truncated=True,num_timeframes=300)


#Person 3: Billie Eilish
person3 = 'billie_eilish'
filter_bona_fide = person3+'_bona-fide'
filter_spoof = person3+'_spoof'
bona_fide_person3 = plot_aggregated_mel_spectrogram(folder_path, filter_bona_fide)
spoof_person3 = plot_aggregated_mel_spectrogram(folder_path, filter_spoof )
plot_melspectrogram(bona_fide_person3, person3, bona_fide=True, truncated=True,num_timeframes=300)
plot_melspectrogram(spoof_person3, person3, bona_fide=False, truncated=True,num_timeframes=300)


