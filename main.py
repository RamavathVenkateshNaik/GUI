import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error
import os
import time

# List of data directories (replace with your actual directories)
data_directories = [
    "/home/sujith/Documents/ML/n1a1data/17L/15K17L/",
    "/home/sujith/Documents/ML/n1a1data/17L/50K17L/",
    "/home/sujith/Documents/ML/n1a1data/17L/77K17L/",
    "/home/sujith/Documents/ML/n1a1data/17L/100K17L/",
    "/home/sujith/Documents/ML/n1a1data/17L/150K17L/",
    "/home/sujith/Documents/ML/n1a1data/17L/200K17L/",
    "/home/sujith/Documents/ML/n1a1data/17L/250K17L/",
    "/home/sujith/Documents/ML/n1a1data/17L/300K17L/"
]

data_directories = data_directories[5:]

# --- Physics-Based Model Function ---
def physical_model(x, a0, a1, n1, tsi, n_max):
    n0 = 1.0  # Fixed constant
    pi_x_tsi = np.pi * x / tsi
    cos2_term = np.cos(a0 * pi_x_tsi) ** 2
    sinh2_term = np.sinh(a1 * pi_x_tsi) ** 2
    return (n0 * cos2_term + n1 * sinh2_term * cos2_term) * n_max

# --- Linear Trend Objective Function ---
def linear_trend_objective(params, x, original_y, tsi, n_max, a0, expected_a1, expected_n1, trend_weight=1000):
    a1, n1 = params
    
    # Primary objective: fitting error
    predicted_y = physical_model(x, a0, a1, n1, tsi, n_max)
    mse = mean_squared_error(original_y, predicted_y)
    
    # Linear trend penalty to ensure smooth increase
    trend_penalty = 0
    if expected_a1 is not None:
        trend_penalty += abs(a1 - expected_a1) * trend_weight
    if expected_n1 is not None:
        trend_penalty += abs(n1 - expected_n1) * trend_weight
    
    return mse + trend_penalty

# Loop through each data directory
for dir_idx, data_dir in enumerate(data_directories):
    print(f"\n\n{'='*50}")
    print(f"Processing data in directory: {data_dir}")
    print(f"{'='*50}")
    
    # Construct file paths
    xchdata_path = os.path.join(data_dir, 'xchdata.csv')
    vg_path = os.path.join(data_dir, 'vg.csv')
    dataprep_path = os.path.join(data_dir, 'dataprep.csv')
    output_csv_path = os.path.join(data_dir, 'dataprep.csv')
    
    # --- Load Data ---
    data = pd.read_csv(xchdata_path, header=None)
    x = data[0].values.reshape(-1, 1)

    # --- Constants ---
    tsi = 2.17   # given

    # --- Load z_data from vg.csv ---
    vg = pd.read_csv(vg_path, delimiter='\t', header=None)
    z_data = vg[0].values  # z values 

    # --- Load existing a0 and n0 values from dataprep.csv ---
    existing_params = pd.read_csv(dataprep_path)
    a0_values = existing_params['a0'].values
    n0_values = existing_params['n0'].values
    
    print(f"Loaded a0 and n0 values from {dataprep_path}")

    # --- Initialize parameter storage ---
    a1_list = []
    n1_list = []

    # --- Initial bounds for a1 and n1 ---
    a1_bounds = (0.000001, 0.1)
    n1_bounds = (0.003936, 0.01024)

    # --- Calculate linear trend targets ---
    num_columns = 21
    
    # For a1: linear increase from min to max bound
    a1_min, a1_max = a1_bounds
    a1_targets = np.linspace(a1_min, a1_max, num_columns)
    
    # For n1: linear increase from min to max bound
    n1_min, n1_max = n1_bounds
    n1_targets = np.linspace(n1_min, n1_max, num_columns)

    # --- Loop through columns Y[1] to Y[21] ---
    for col in range(1, 22):
        print(f"\n{'='*20} Processing Y[{col}] {'='*20}")
        
        original_y = data[col].values
        n_max = max(original_y)
        
        # Get a0 and n0 from existing data
        a0 = a0_values[col-1]  # col-1 because columns are 1-indexed but arrays are 0-indexed
        n0 = n0_values[col-1]
        
        print(f"Column {col}: Using a0 = {a0:.6f}, n0 = {n0:.6f}")
        
        # Get expected linear trend values
        expected_a1 = a1_targets[col-1]
        expected_n1 = n1_targets[col-1]
        
        print(f"Target a1 = {expected_a1:.6f}, Target n1 = {expected_n1:.6f}")
        
        # Set bounds around expected values with some tolerance
        tolerance_a1 = (a1_max - a1_min) * 0.1  # 10% tolerance
        tolerance_n1 = (n1_max - n1_min) * 0.1  # 10% tolerance
        
        bounds = [
            (max(a1_min, expected_a1 - tolerance_a1), min(a1_max, expected_a1 + tolerance_a1)),
            (max(n1_min, expected_n1 - tolerance_n1), min(n1_max, expected_n1 + tolerance_n1))
        ]
        
        print(f"Optimization bounds: a1={bounds[0]}, n1={bounds[1]}")
        
        # Run optimization
        result = differential_evolution(
            linear_trend_objective,
            bounds=bounds,
            args=(x, original_y, tsi, n_max, a0, expected_a1, expected_n1),
            strategy='best1bin',
            maxiter=400,
            tol=1e-8,
            polish=True,
            seed=42  # For reproducibility
        )
        
        best_a1, best_n1 = result.x
        a1_list.append(best_a1)
        n1_list.append(best_n1)
        
        predicted_y = np.squeeze(physical_model(x, a0, best_a1, best_n1, tsi, n_max))
        
        print("\nOptimized parameters:")
        print(f"a0 = {a0:.6f} (from existing data)")
        print(f"a1 = {best_a1:.6f}")
        print(f"n0 = {n0:.6f} (from existing data)")
        print(f"n1 = {best_n1:.6f}")
        
        # --- Plot: Original, Predicted ---
        plt.figure(figsize=(10, 6))
        plt.plot(x, original_y, 'o-', label='Original Y', color='black')
        plt.plot(x, predicted_y, 's--', label='Predicted Y', color='red')
        plt.title(f'Y[{col}]: Original vs Predicted')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.5)  # Show plot for 0.5 seconds
        plt.close()

    # Plot parameter variations
    plt.figure(figsize=(10, 4))

    # Plot a1 vs z
    plt.subplot(1, 2, 1)
    plt.plot(z_data, a1_list, 'o-', color='green')
    plt.title('a1 vs z (Linear Trend)')
    plt.xlabel('z')
    plt.ylabel('a1')
    plt.xticks(np.linspace(min(z_data), max(z_data), 6))
    plt.grid(True)

    # Plot n1 vs z
    plt.subplot(1, 2, 2)
    plt.plot(z_data, n1_list, 'x-', color='purple')
    plt.title('n1 vs z (Linear Trend)')
    plt.xlabel('z')
    plt.ylabel('n1')
    plt.xticks(np.linspace(min(z_data), max(z_data), 6))
    plt.grid(True)

    plt.tight_layout()
    # Save the parameter patterns plot in the current data directory
    parameters_pattern_filename = os.path.join(data_dir, 'Linear_Trend_a1_n1.png')
    plt.savefig(parameters_pattern_filename, dpi=300)
    print(f"Parameter patterns plot saved as {parameters_pattern_filename}")
    plt.close()

    # Update existing dataprep.csv with optimized a1 and n1 values
    existing_params['a1'] = [round(val, 6) for val in a1_list]
    existing_params['n1'] = [round(val, 6) for val in n1_list]

    # Save the updated CSV file with all existing columns plus optimized a1 and n1
    existing_params.to_csv(output_csv_path, index=False)
    
    print(f"Updated dataprep.csv with optimized a1 and n1 parameters: '{output_csv_path}'")

print("\n\n" + "="*50)
print("PROCESSING COMPLETE")
print("="*50)
print("All datasets processed with linear trend optimization for a1 and n1")
print("a0 and n0 values extracted from existing dataprep.csv files")
print("Existing dataprep.csv files updated with optimized a1 and n1 parameters")
