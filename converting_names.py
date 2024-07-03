import pandas as pd
import os

# Define the folder path and CSV file path
folder_path = "./release_in_the_wild/"
csv_file_path = "./renaming_file.csv"

mapping = pd.read_csv(csv_file_path, names=["old_name", "speaker", "label", "new_name"])

# Rename files according to the mapping
for i, row in mapping.iterrows():
    old_file_path = os.path.join(folder_path, row['old_name'])
    if os.path.exists(old_file_path):
        new_file_path = os.path.join(folder_path, row['new_name'])
        os.rename(old_file_path, new_file_path)
        print(f"Renamed '{row['old_name']}' to '{row['new_name']}'")
    else:
        print(f"No mapping found for '{row['old_name']}', skipping...")
