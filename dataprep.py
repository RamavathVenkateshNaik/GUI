import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error
import os
import time
import re

# List of data directories (replace with your actual directories)
data_directories = [
    "/home/sujith/Documents/ML/dataprep/75L/15K75L/",
    "/home/sujith/Documents/ML/dataprep/75L/50K75L/",
    "/home/sujith/Documents/ML/dataprep/75L/77K75L/",
    "/home/sujith/Documents/ML/dataprep/75L/100K75L/",
    "/home/sujith/Documents/ML/dataprep/75L/150K75L/",
    "/home/sujith/Documents/ML/dataprep/75L/200K75L/",
    "/home/sujith/Documents/ML/dataprep/75L/250K75L/",
    "/home/sujith/Documents/ML/dataprep/75L/300K75L/"
]

data_directories = data_directories[5:]

# Function to read a0 values from a0_summary.txt
def read_a0_values(filename='a0_summary.txt'):
    """Read a0_avg values from the summary file"""
    a0_values = []
    
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found. Using default a0 values.")
        # Return default values if file doesn't exist
        return [0.815] * len(data_directories)
    
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            # Look for lines containing "a0_avg = "
            if "a0_avg = " in line:
                # Extract the numerical value after "a0_avg = "
                match = re.search(r'a0_avg = ([\d.]+)', line)
                if match:
                    a0_values.append(float(match.group(1)))
        
        # If we found the overall average line, remove it (it's not per dataset)
        if len(a0_values) > len(data_directories):
            a0_values = a0_values[:len(data_directories)]
            
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        print("Using default a0 values.")
        return [0.815] * len(data_directories)
    
    return a0_values

# Read a0 values from the summary file
a0_values_from_file = read_a0_values()

print("Loaded a0 values from a0_summary.txt:")
for i, a0_val in enumerate(a0_values_from_file):
    print(f"  Dataset {i+1}: a0 = {a0_val:.6f}")

# Loop through each data directory
for dir_idx, data_dir in enumerate(data_directories):
    print(f"\n\n{'='*50}")
    print(f"Processing data in directory: {data_dir}")
    print(f"{'='*50}")
    
    # Get the fixed a0 value for this directory
    fixed_a0 = a0_values_from_file[dir_idx] if dir_idx < len(a0_values_from_file) else 0.815
    print(f"Using fixed a0 value: {fixed_a0:.6f}")
    
    # Construct file paths
    xchdata_path = os.path.join(data_dir, 'xchdata.csv')
    vg_path = os.path.join(data_dir, 'vg.csv')
    output_csv_path = os.path.join(data_dir, 'dataprep.csv')
    
    # --- Load Data ---
    data = pd.read_csv(xchdata_path, header=None)
    x = data[0].values.reshape(-1, 1)

    # --- Constants ---
    tsi = 10.0455    # given

    # --- Load z_data from vg.csv ---
    vg = pd.read_csv(vg_path, delimiter='\t', header=None)
    z_data = vg[0].values  # z values 

    # --- Find index where x is closest to 0 (to determine nc) ---
    x_0_idx = np.argmin(np.abs(x))

    # --- Initialize parameter storage ---
    a0_list = []  # Will store the fixed a0 value for each column
    a1_list = []
    n1_list = []
    n0_list = []  # We'll calculate this directly but still track it

    # --- Physics-Based Model Function (with n0 calculated, not optimized) ---
    def physical_model(x, a0, a1, n0, n1, tsi, n_max):
        pi_x_tsi = np.pi * x / tsi
        cos2_term = np.cos(a0 * pi_x_tsi) ** 2
        sinh2_term = np.sinh(a1 * pi_x_tsi) ** 2
        return (n0 * cos2_term + n1 * sinh2_term * cos2_term) * n_max

    # --- Objective Function for Optimization (MSE) - Only optimizing a1 and n1 ---
    def objective(params, x, original_y, tsi, n_max, n0, fixed_a0):
        a1, n1 = params  # Only a1 and n1 are being optimized
        predicted_y = physical_model(x, fixed_a0, a1, n0, n1, tsi, n_max)
        return mean_squared_error(original_y, predicted_y)

    # --- Parameter Bounds - Only for a1 and n1 now ---
    bounds = [
        (0.003346, 3),      # a1
        (0.000000, 1)      # n1
    ]

    # --- Loop through columns Y[1] to Y[21] ---
    for col in range(1, 22):
        print(f"\n{'='*20} Processing Y[{col}] {'='*20}")
        
        original_y = data[col].values
        n_max = max(original_y)
        
        # Calculate n0 directly from the data: n0 = nc/nmax
        nc = original_y[x_0_idx]  # value at x closest to 0
        n0 = nc / n_max
        n0_list.append(n0)
        a0_list.append(fixed_a0)  # Store the fixed a0 value
        print(f"Column {col}: nc = {nc}, n_max = {n_max}, n0 = {n0:.6f}, a0 = {fixed_a0:.6f} (fixed)")
        
        # Define a callback function to track convergence at high precision
        param_history = []
        def track_params(xk, convergence):
            param_history.append(xk)
            # Print current best parameters with 6 decimal places precision
            if len(param_history) % 50 == 0:  # Print every 50 iterations to avoid cluttering
                print(f"Current best: a1={xk[0]:.6f}, n1={xk[1]:.6f}, convergence={convergence:.8f}")

        # Modified differential evolution call with enhanced precision settings
        result = differential_evolution(
            objective,
            bounds=bounds,
            args=(x, original_y, tsi, n_max, n0, fixed_a0),  # Pass fixed_a0 as argument
            callback=track_params,
            strategy='best1bin',
            maxiter=800,       # Increased from 500 to ensure convergence
            popsize=20,         # Increased population size for better exploration
            tol=1e-8,           # Tightened tolerance from 1e-6 to 1e-8 for higher precision
            mutation=(0.5, 1.0),# Adaptive mutation for fine-grained search
            recombination=0.7,  # Slightly increased from default for better mixing
            polish=True,        # Final polishing with local optimizer
            updating='deferred', # Better for precision-critical applications
            workers=1          # Use all available cores for parallel processing
        )
        
        best_a1, best_n1 = result.x  # Only a1 and n1 are optimized
        a1_list.append(best_a1)
        n1_list.append(best_n1)
        
        predicted_y = np.squeeze(physical_model(x, fixed_a0, best_a1, n0, best_n1, tsi, n_max))
        
        print("\nBest-fit parameters:")
        print(f"a0 = {fixed_a0:.6f} (fixed)")
        print(f"a1 = {best_a1:.6f}")
        print(f"n0 = {n0:.6f} (calculated directly, not optimized)")
        print(f"n1 = {best_n1:.6f}")
        
        # --- Plot: Original, Predicted, and Internal Terms ---
        plt.figure(figsize=(10, 6))
        plt.plot(x, original_y, 'o-', label='Original Y', color='black')
        plt.plot(x, predicted_y, 's--', label='Predicted Y', color='red')
        plt.title(f'Y[{col}]: Original, Predicted, and Term Contributions')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(1)  # Show plot for 0.5 seconds
        plt.close()

    # Plot parameter variations
    plt.figure(figsize=(15, 4))

    # Plot a1 vs z
    plt.subplot(1, 4, 1)
    plt.plot(z_data, a1_list, 'o-', color='blue')
    plt.title('a1 vs z')
    plt.xlabel('z')
    plt.ylabel('a1')
    plt.xticks(np.linspace(min(z_data), max(z_data), 6))
    plt.grid(True)

    # Plot n0 vs z
    plt.subplot(1, 4, 2)
    plt.plot(z_data, n0_list, 's-', color='green')
    plt.title('n0 vs z')
    plt.xlabel('z')
    plt.ylabel('n0')
    plt.xticks(np.linspace(min(z_data), max(z_data), 6))
    plt.grid(True)

    # Plot n1 vs z
    plt.subplot(1, 4, 3)
    plt.plot(z_data, n1_list, 'd-', color='red')
    plt.title('n1 vs z')
    plt.xlabel('z')
    plt.ylabel('n1')
    plt.xticks(np.linspace(min(z_data), max(z_data), 6))
    plt.grid(True)

    # Plot a0 vs z (will be constant for each dataset)
    plt.subplot(1, 4, 4)
    plt.plot(z_data, a0_list, 'x-', color='pink')
    plt.title('a0 vs z')
    plt.xlabel('z')
    plt.ylabel('a0')
    plt.xticks(np.linspace(min(z_data), max(z_data), 6))
    plt.grid(True)

    plt.tight_layout()
    # Save the parameter patterns plot in the current data directory
    parameters_pattern_filename = os.path.join(data_dir, 'parameters_pattern.png')
    plt.savefig(parameters_pattern_filename, dpi=300)
    print(f"Parameter patterns plot saved as {parameters_pattern_filename}")
    plt.close()
    
    # Calculate the average value of a0_list (will be same as fixed_a0)
    average_a0 = np.mean(a0_list)
    print(f"\nAverage value of a0 for {data_dir}: {average_a0:.6f} (fixed)")

    # Save parameters to CSV
    params_df = pd.DataFrame({
        'z': z_data,
        'a1': a1_list,
        'n0': n0_list,
        'n1': n1_list,
        'a0': a0_list
    })
     
    params_df['a0_avg'] = average_a0

    # Round all numeric columns to 6 decimal places
    for col in params_df.columns:
        if col != 'z':  # Keep z as is if it's not numeric or should not be rounded
            params_df[col] = params_df[col].round(6)

    # Save the CSV file with parameters
    params_df.to_csv(output_csv_path, index=False)
    
    print(f"Parameters saved to '{output_csv_path}' with a0_avg as a column")

print("\n\n" + "="*50)
print("PROCESSING COMPLETE")
print("="*50)
print("All datasets processed with fixed a0 values from a0_summary.txt")
print("Only a1 and n1 were optimized through differential evolution")
print("All parameters (including fixed a0) saved to fitted_parameters.csv in each directory")
