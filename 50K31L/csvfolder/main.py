import os
import pandas as pd

# Path to your folder
folder_path = '/home/sujith/Documents/ML/31L/50K31L/csvfolder'
save_path = '/home/sujith/Documents/ML/31L/50K31L'
# Collect all .dat files
dat_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.dat')])

# Initialize list to collect each file's data
columns = []

# Read and store data from each .dat file
for dat_file in dat_files:
    file_path = os.path.join(folder_path, dat_file)
    data = pd.read_csv(file_path, header=None)
    columns.append(data[0])

# Combine all columns (each as Series) into a DataFrame
combined_df = pd.concat(columns, axis=1)

# Insert an empty first column (all NaNs)
combined_df.insert(0, '', '')

# Save without header or index
output_path = os.path.join(save_path, 'xchdata.csv')
combined_df.to_csv(output_path, index=False, header=False)

print(f"Combined {len(dat_files)} files with an empty first column into {output_path}")

