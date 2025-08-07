import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error
import os
import time

# List of data directories (replace with your actual directories)
data_directories = [
    "/home/sujith/Documents/ML/31L/300K31L/",
    "/home/sujith/Documents/ML/31L/250K31L/",
    "/home/sujith/Documents/ML/31L/200K31L/",
    "/home/sujith/Documents/ML/31L/150K31L/"
]

# Initialize storage for a0_avg values across all datasets
all_a0_avg_values = []

# Loop through each data directory
for dir_idx, data_dir in enumerate(data_directories):
    print(f"\n\n{'='*50}")
    print(f"Processing data in directory: {data_dir}")
    print(f"{'='*50}")
    
    # Construct file paths
    xchdata_path = os.path.join(data_dir, 'xchdata.csv')
    vg_path = os.path.join(data_dir, 'vg.csv')
    output_csv_path = os.path.join(data_dir, 'fitted_parameters.csv')
    
    # --- Load Data ---
    data = pd.read_csv(xchdata_path, header=None)
    x = data[0].values.reshape(-1, 1)

    # --- Constants ---
    tsi = 4.0732    # given

    # --- Load z_data from vg.csv ---
    vg = pd.read_csv(vg_path, delimiter='\t', header=None)
    z_data = vg[0].values  # z values 

    # --- Find index where x is closest to 0 (to determine nc) ---
    x_0_idx = np.argmin(np.abs(x))

    # --- Initialize parameter storage ---
    a0_list = []
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
    def objective(params, x, original_y, tsi, n_max, n0):
        a0, a1, n1 = params
        predicted_y = physical_model(x, a0, a1, n0, n1, tsi, n_max)
        return mean_squared_error(original_y, predicted_y)

    # --- Parameter Bounds - Only for a1 and n1 now ---
    bounds = [
        (0.85, 1),      # a0
        (0.0000346, 3),      # a1
        (0.0000346, 1)    # n1
    ]

    # --- Loop through columns Y[1] to Y[24] ---
    for col in range(1, 22):
        print(f"\n{'='*20} Processing Y[{col}] {'='*20}")
        
        original_y = data[col].values
        n_max = max(original_y)
        
        # Calculate n0 directly from the data: n0 = nc/nmax
        nc = original_y[x_0_idx]  # value at x closest to 0
        n0 = nc / n_max
        n0_list.append(n0)
        print(f"Column {col}: nc = {nc}, n_max = {n_max}, n0 = {n0:.6f}")
        
        param_history = []
        def track_params(xk, convergence):
            param_history.append(xk)
        
        result = differential_evolution(
            objective,
            bounds=bounds,
            args=(x, original_y, tsi, n_max, n0),  # Pass n0 as a fixed parameter
            callback=track_params,
            strategy='best1bin',
            maxiter=200,
            tol=1e-6,
            polish=True
        )
        
        best_a0, best_a1, best_n1 = result.x
        a0_list.append(best_a0)
        a1_list.append(best_a1)
        n1_list.append(best_n1)
        
        predicted_y = np.squeeze(physical_model(x, best_a0, best_a1, n0, best_n1, tsi, n_max))
        
        print("\nBest-fit parameters:")
        print(f"a1 = {best_a1:.6f}")
        print(f"n0 = {n0:.6f} (calculated directly, not optimized)")
        print(f"n1 = {best_n1:.6f}")
        print(f"a0 = {best_a0:.6f}")
        
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
        plt.pause(0.5)  # Show plot for 0.5 seconds
        plt.close()

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

    # Plot a0 vs z
    plt.subplot(1, 4, 4)
    plt.plot(z_data, a0_list, 'x-', color='pink')
    plt.title('a0 vs z')
    plt.xlabel('z')
    plt.ylabel('a0')
    plt.xticks(np.linspace(min(z_data), max(z_data), 6))
    plt.grid(True)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)  # Show plot for 0.5 seconds
    plt.close()

    # Calculate the average value of a0_list
    average_a0 = np.mean(a0_list)
    all_a0_avg_values.append(average_a0)
    print(f"\nAverage value of a0 for {data_dir}: {average_a0:.6f}")

    # Save parameters to CSV (without a0_avg as a column)
    params_df = pd.DataFrame({
        'z': z_data,
        'a1': a1_list,
        'n0': n0_list,
        'n1': n1_list,
        'a0': a0_list
    })
    
    # Save the CSV file with parameters
    params_df.to_csv(output_csv_path, index=False)
    
    # Append a0_avg as a single entry at the end of the file
    with open(output_csv_path, 'a') as f:
        f.write(f"\na0_avg,{average_a0:.6f}\n")
    
    print(f"Parameters saved to '{output_csv_path}' with a0_avg appended")

# Print summary of a0_avg values across all datasets
print("\n\n" + "="*50)
print("SUMMARY OF a0_avg VALUES ACROSS ALL DATASETS")
print("="*50)
for idx, data_dir in enumerate(data_directories):
    print(f"Dataset {idx+1} ({data_dir}): a0_avg = {all_a0_avg_values[idx]:.6f}")

# Calculate overall average a0 value
overall_avg_a0 = np.mean(all_a0_avg_values)
print("\nOverall average a0 value across all datasets: {:.6f}".format(overall_avg_a0))

# After processing all datasets, create a simple summary file
with open('a0_summary.txt', 'w') as f:
    for idx, data_dir in enumerate(data_directories):
        f.write(f"Dataset {idx+1} ({data_dir}): a0_avg = {all_a0_avg_values[idx]:.6f}\n")
    f.write(f"\nOverall average a0 value: {overall_avg_a0:.6f}\n")

print("Summary of a0_avg values saved to 'a0_summary.txt'")
