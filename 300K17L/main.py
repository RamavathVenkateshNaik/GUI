import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error

# --- Load Data ---
data = pd.read_csv('xchdata.csv', header=None)
x = data[0].values.reshape(-1, 1)

# --- Constants ---
tsi = 2.1724  # fixed

# --- Load z_data ---
vg = pd.read_csv('vg.csv', delimiter='\t', header=None)
z_data = vg[0].values

# --- Initialize parameter storage ---
a0_list = []
a1_list = []
n0_list = []
n1_list = []

# --- Physics-Based Model Function ---
def physical_model(x, a0, a1, n1, tsi, n_max):
    n0 = 1.0  # Fixed constant
    pi_x_tsi = np.pi * x / tsi
    cos2_term = np.cos(a0 * pi_x_tsi) ** 2
    sinh2_term = np.sinh(a1 * pi_x_tsi) ** 2
    return (n0 * cos2_term + n1 * sinh2_term * cos2_term) * n_max

# --- Custom Objective Function with Smoothness Penalty ---
def custom_objective(params, x, original_y, tsi, n_max, prev_params, smoothness_weight=100):
    a0, a1, n1 = params
    
    # Primary objective: fitting error
    predicted_y = physical_model(x, a0, a1, n1, tsi, n_max)
    mse = mean_squared_error(original_y, predicted_y)
    
    # Smoothness penalty
    penalty = 0
    if prev_params is not None:
        prev_a0, prev_a1, prev_n1 = prev_params
        penalty += abs(a0 - prev_a0)
        penalty += abs(a1 - prev_a1) * 0.1  # a1 has less weight
        penalty += abs(n1 - prev_n1)
    
    return mse + smoothness_weight * penalty

# --- Initial bounds: (a0, a1, n1) ---
initial_bounds = [
    (0.7, 1),           # a0
    (0.000001, 0.0004), # a1
    (0.03936, 0.1024)   # n1
]

# --- Adaptive bounds parameters ---
bound_reduction_factor = 0.8
min_bound_width = 0.01

# --- Store previous parameters for smoothness ---
prev_params = None

# --- Loop through columns Y[1] to Y[21] ---
for col in range(1, 22):
    print(f"\n{'='*20} Processing Y[{col}] {'='*20}")
    original_y = data[col].values
    n_max = max(original_y)
    
    # Use adaptive bounds
    if col == 1:
        bounds = initial_bounds.copy()
    else:
        # Adapt bounds based on previous result
        prev_a0, prev_a1, prev_n1 = prev_params
        
        # Calculate new bounds with reduction
        a0_width = max(min_bound_width, (bounds[0][1] - bounds[0][0]) * bound_reduction_factor)
        a1_width = max(min_bound_width * 0.0001, (bounds[1][1] - bounds[1][0]) * bound_reduction_factor)
        n1_width = max(min_bound_width, (bounds[2][1] - bounds[2][0]) * bound_reduction_factor)
        
        # Ensure a0 increases slowly
        a0_min = max(prev_a0, bounds[0][0])
        a0_max = min(prev_a0 + a0_width, bounds[0][1])
        
        # Ensure a1 increases slowly
        a1_min = max(prev_a1, bounds[1][0])
        a1_max = min(prev_a1 + a1_width, bounds[1][1])
        
        # n1 decreasing trend
        n1_min = max(bounds[2][0], prev_n1 - n1_width)
        n1_max = min(prev_n1, bounds[2][1])
        
        bounds = [
            (a0_min, a0_max),
            (a1_min, a1_max),
            (n1_min, n1_max)
        ]
    
    print(f"Current bounds: {bounds}")
    
    # Run optimization
    result = differential_evolution(
        custom_objective,
        bounds=bounds,
        args=(x, original_y, tsi, n_max, prev_params),
        strategy='best1bin',
        maxiter=400,
        tol=1e-8,
        polish=True,
        seed=42  # For reproducibility
    )
    
    best_a0, best_a1, best_n1 = result.x
    
    # Store parameters
    a0_list.append(best_a0)
    a1_list.append(best_a1)
    n0_list.append(1.0)  # n0 is constant
    n1_list.append(best_n1)
    
    # Update previous parameters for next iteration
    prev_params = (best_a0, best_a1, best_n1)
    
    predicted_y = np.squeeze(physical_model(x, best_a0, best_a1, best_n1, tsi, n_max))
    
    print("\nBest-fit parameters:")
    print(f"a0 = {best_a0:.6f}")
    print(f"a1 = {best_a1:.6f}")
    print(f"n0 = 1.000000 (fixed)")
    print(f"n1 = {best_n1:.6f}")
    
    # --- Plot: Original and Predicted ---
    plt.figure(figsize=(10, 6))
    plt.plot(x, original_y, 'o-', label='Original Y', color='black')
    plt.plot(x, predicted_y, 's--', label='Predicted Y', color='red')
    plt.title(f'Y[{col}]: Original vs Predicted')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Plot parameter evolution ---
plt.figure(figsize=(15, 4))

# Plot a0 vs z (should increase)
plt.subplot(1, 4, 1)
plt.plot(z_data, a0_list, 'o-', color='blue')
plt.title('a0 vs z (increasing)')
plt.xlabel('z')
plt.ylabel('a0')
plt.grid(True)

# Plot a1 vs z (should increase slowly)
plt.subplot(1, 4, 2)
plt.plot(z_data, a1_list, 's-', color='green')
plt.title('a1 vs z (increasing slowly)')
plt.xlabel('z')
plt.ylabel('a1')
plt.grid(True)

# Plot n0 vs z (should stay close to 1)
plt.subplot(1, 4, 3)
plt.plot(z_data, n0_list, 'd-', color='red')
plt.title('n0 vs z (close to 1)')
plt.xlabel('z')
plt.ylabel('n0')
plt.grid(True)

# Plot n1 vs z (should decrease)
plt.subplot(1, 4, 4)
plt.plot(z_data, n1_list, 'x-', color='purple')
plt.title('n1 vs z (decreasing)')
plt.xlabel('z')
plt.ylabel('n1')
plt.grid(True)

plt.tight_layout()
plt.show()
