import os
import librosa
import numpy as np
import common_functions

########## creating numpy files to the original files
folder_path = './release_in_the_wild'
nparray_files = './mfccs_files'
if not os.path.exists(nparray_files):
    os.mkdir(nparray_files)


    
durations_origin = []
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    mfcc, duration = extract_features_mfcc(file_path)
    if duration < 1:
        continue
    durations_origin.append(duration)
    file_name_no_ext = os.path.splitext(file_name)[0]
    new_file_path = os.path.join(nparray_files, file_name_no_ext + '.npy')
    # Save the numpy array to the file
    np.save(new_file_path, mfcc)
np.save(nparray_files + '_durations.npy', durations_origin)

########## creating numpy files to the cleaned and padded files

folder_path_clean = './release_in_the_wild_clean'
nparray_files_clean = './mfccs_files_clean'
if not os.path.exists(nparray_files_clean):
    os.mkdir(nparray_files_clean)

for file_name in os.listdir(folder_path_clean):
    file_path = os.path.join(folder_path_clean, file_name)
    mfcc, _ = extract_features_mfcc(file_path)
    file_name_no_ext = os.path.splitext(file_name)[0]
    new_file_path = os.path.join(nparray_files_clean, file_name_no_ext + '.npy')
    # Save the numpy array to the file
    np.save(new_file_path, mfcc)


