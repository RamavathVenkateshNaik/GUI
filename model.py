import numpy as np
import pandas as pd
import os

# Load all dataprep.csv into a single DataFrame
data_directories  = [
    r"C:\Users\acer\Desktop\web\n1a1data\17L\15K17L",
    r"C:\Users\acer\Desktop\web\n1a1data\17L\50K17L",
    r"C:\Users\acer\Desktop\web\n1a1data\17L\77K17L",
    r"C:\Users\acer\Desktop\web\n1a1data\17L\100K17L",
    r"C:\Users\acer\Desktop\web\n1a1data\17L\150K17L",
    r"C:\Users\acer\Desktop\web\n1a1data\17L\200K17L",
    r"C:\Users\acer\Desktop\web\n1a1data\17L\250K17L",
    r"C:\Users\acer\Desktop\web\n1a1data\17L\300K17L",
    r"C:\Users\acer\Desktop\web\n1a1data\23L\15K23L",
    r"C:\Users\acer\Desktop\web\n1a1data\23L\50K23L",
    r"C:\Users\acer\Desktop\web\n1a1data\23L\77K23L",
    r"C:\Users\acer\Desktop\web\n1a1data\23L\100K23L",
    r"C:\Users\acer\Desktop\web\n1a1data\23L\150K23L",
    r"C:\Users\acer\Desktop\web\n1a1data\23L\200K23L",
    r"C:\Users\acer\Desktop\web\n1a1data\23L\250K23L",
    r"C:\Users\acer\Desktop\web\n1a1data\23L\300K23L",
    r"C:\Users\acer\Desktop\web\n1a1data\31L\15K31L",
    r"C:\Users\acer\Desktop\web\n1a1data\31L\50K31L",
    r"C:\Users\acer\Desktop\web\n1a1data\31L\77K31L",
    r"C:\Users\acer\Desktop\web\n1a1data\31L\100K31L",
    r"C:\Users\acer\Desktop\web\n1a1data\31L\150K31L",
    r"C:\Users\acer\Desktop\web\n1a1data\31L\200K31L",
    r"C:\Users\acer\Desktop\web\n1a1data\31L\250K31L",
    r"C:\Users\acer\Desktop\web\n1a1data\31L\300K31L",
    r"C:\Users\acer\Desktop\web\n1a1data\37L\15K37L",
    r"C:\Users\acer\Desktop\web\n1a1data\37L\50K37L",
    r"C:\Users\acer\Desktop\web\n1a1data\37L\77K37L",
    r"C:\Users\acer\Desktop\web\n1a1data\37L\100K37L",
    r"C:\Users\acer\Desktop\web\n1a1data\37L\150K37L",
    r"C:\Users\acer\Desktop\web\n1a1data\37L\200K37L",
    r"C:\Users\acer\Desktop\web\n1a1data\37L\250K37L",
    r"C:\Users\acer\Desktop\web\n1a1data\37L\300K37L",
    r"C:\Users\acer\Desktop\web\n1a1data\53L\15K53L",
    r"C:\Users\acer\Desktop\web\n1a1data\53L\50K53L",
    r"C:\Users\acer\Desktop\web\n1a1data\53L\77K53L",
    r"C:\Users\acer\Desktop\web\n1a1data\53L\100K53L",
    r"C:\Users\acer\Desktop\web\n1a1data\53L\150K53L",
    r"C:\Users\acer\Desktop\web\n1a1data\53L\200K53L",
    r"C:\Users\acer\Desktop\web\n1a1data\53L\250K53L",
    r"C:\Users\acer\Desktop\web\n1a1data\53L\300K53L",
    r"C:\Users\acer\Desktop\web\n1a1data\61L\15K61L",
    r"C:\Users\acer\Desktop\web\n1a1data\61L\50K61L",
    r"C:\Users\acer\Desktop\web\n1a1data\61L\77K61L",
    r"C:\Users\acer\Desktop\web\n1a1data\61L\100K61L",
    r"C:\Users\acer\Desktop\web\n1a1data\61L\150K61L",
    r"C:\Users\acer\Desktop\web\n1a1data\61L\200K61L",
    r"C:\Users\acer\Desktop\web\n1a1data\61L\250K61L",
    r"C:\Users\acer\Desktop\web\n1a1data\61L\300K61L",
    r"C:\Users\acer\Desktop\web\n1a1data\69L\15K69L",
    r"C:\Users\acer\Desktop\web\n1a1data\69L\50K69L",
    r"C:\Users\acer\Desktop\web\n1a1data\69L\77K69L",
    r"C:\Users\acer\Desktop\web\n1a1data\69L\100K69L",
    r"C:\Users\acer\Desktop\web\n1a1data\69L\150K69L",
    r"C:\Users\acer\Desktop\web\n1a1data\69L\200K69L",
    r"C:\Users\acer\Desktop\web\n1a1data\69L\250K69L",
    r"C:\Users\acer\Desktop\web\n1a1data\69L\300K69L",
    r"C:\Users\acer\Desktop\web\n1a1data\75L\15K75L",
    r"C:\Users\acer\Desktop\web\n1a1data\75L\50K75L",
    r"C:\Users\acer\Desktop\web\n1a1data\75L\77K75L",
    r"C:\Users\acer\Desktop\web\n1a1data\75L\100K75L",
    r"C:\Users\acer\Desktop\web\n1a1data\75L\150K75L",
    r"C:\Users\acer\Desktop\web\n1a1data\75L\200K75L",
    r"C:\Users\acer\Desktop\web\n1a1data\75L\250K75L",
    r"C:\Users\acer\Desktop\web\n1a1data\75L\300K75L"
]  

all_data = []
for directory in data_directories:
    csv_path = os.path.join(directory, "dataprep.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Extract layer and temperature from folder name
        folder = os.path.basename(os.path.normpath(directory))
        temp_str, layer_str = folder.split('K')
        temp = int(temp_str)
        layer = int(layer_str[:-1])
        tsi = (layer - 1) * 0.13575  # nanometer
        df['temperature'] = temp
        df['num_layers'] = layer
        df['tsi'] = tsi
        all_data.append(df)

combined_df = pd.concat(all_data, ignore_index=True)

def lookup_parameters(voltage, temperature, tsi):
    """
    Interpolate n0, n1, a0, a1, nmax from combined_df based on user input.
    """
    from scipy.interpolate import griddata
    
    points = combined_df[['z', 'temperature', 'tsi']].values
    n0_values = combined_df['n0'].values
    n1_values = combined_df['n1'].values
    a0_values = combined_df['a0'].values
    a1_values = combined_df['a1'].values
    nmax_values = combined_df['nmax'].values
    
    query = np.array([[voltage, temperature, tsi]])
    
    n0 = griddata(points, n0_values, query, method='linear')[0]
    n1 = griddata(points, n1_values, query, method='linear')[0]
    a0 = griddata(points, a0_values, query, method='linear')[0]
    a1 = griddata(points, a1_values, query, method='linear')[0]
    nmax = griddata(points, nmax_values, query, method='linear')[0]
    
    return n0, n1, a0, a1, nmax

def physical_model(x, n0, n1, a0, a1, tsi, nmax):
    """Calculate carrier concentration along channel."""
    pi_x_tsi = np.pi * x / tsi
    return nmax * (n0 + n1 * np.sinh(a1 * pi_x_tsi)**2) * (np.cos(a0 * pi_x_tsi)**2)

def calculate_x(tsi, num_points=200):
    """Generate x-axis values along channel thickness."""
    return np.linspace(-tsi/2, tsi/2, num_points)
