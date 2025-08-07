import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error

# --- Load Data ---
data = pd.read_csv('xchdata.csv', header=None)
x = data[0].values.reshape(-1, 1)

# --- Constants ---
tsi = 4.8879  # fixed

# --- Load z_data ---
vg = pd.read_csv('vg.csv', delimiter='\t', header=None)
z_data = vg[0].values

# --- Initialize parameter storage ---
a0_list = []
a1_list = []
n0_list = []
n1_list = []

# --- Physics-Based Model Function ---
def physical_model(x, a0, a1, n0, n1, tsi, n_max):
    pi_x_tsi = np.pi * x / tsi
    cos2_term = np.cos(a0 * pi_x_tsi) ** 2
    sinh2_term = np.sinh(a1 * pi_x_tsi) ** 2
    return (n0 * cos2_term + n1 * sinh2_term * cos2_term) * n_max

# --- Objective Function ---
def objective(params, x, original_y, tsi, n_max):
    a0, a1, n0, n1 = params
    predicted_y = physical_model(x, a0, a1, n0, n1, tsi, n_max)
    return mean_squared_error(original_y, predicted_y)

# --- Bounds: (a0, a1, n0, n1) ---
bounds = [
    (0.85,1),      # a0
    (0.0000346,3),    # a1
    (0.0000346,1),    # n0
    (0.0000346,1)  # n1
]

# --- Loop through columns Y[1] to Y[24] ---
for col in range(1, 22):
    print(f"\n{'='*20} Processing Y[{col}] {'='*20}")
    original_y = data[col].values
    n_max = max(original_y)

    result = differential_evolution(
        objective,
        bounds=bounds,
        args=(x, original_y, tsi, n_max),
        strategy='best1bin',
        maxiter=300,
        tol=1e-8,
        polish=True
    )

    best_a0, best_a1, best_n0, best_n1 = result.x
    a0_list.append(best_a0)
    a1_list.append(best_a1)
    n0_list.append(best_n0)
    n1_list.append(best_n1)

    predicted_y = np.squeeze(physical_model(x, best_a0, best_a1, best_n0, best_n1, tsi, n_max))

    print("\nBest-fit parameters:")
    print(f"a0 = {best_a0:.6f}")
    print(f"a1 = {best_a1:.6f}")
    print(f"n0 = {best_n0:.6f}")
    print(f"n1 = {best_n1:.6f}")

    # --- Internal terms ---
    pi_x_tsi = np.pi * x / tsi
    cos2_term = np.cos(best_a0 * pi_x_tsi) ** 2
    sinh2_term = np.sinh(best_a1 * pi_x_tsi) ** 2
    term1 = best_n0 * cos2_term
    term2 = best_n1 * sinh2_term * cos2_term

       # --- Plot: Original, Predicted, and Internal Terms ---
    plt.figure(figsize=(10, 6))
    plt.plot(x, original_y, 'o-', label='Original Y', color='black')
    plt.plot(x, predicted_y, 's--', label='Predicted Y', color='red')
    #plt.plot(x, term1, '--', label=r'$n_0 \cdot \cos^2$', color='blue')
    #plt.plot(x, term2, '--', label=r'$n_1 \cdot \sinh^2 \cdot \cos^2$', color='green')
    plt.title(f'Y[{col}]: Original, Predicted, and Term Contributions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


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
plt.plot(z_data, a0_list, 'x-', color='pink')  # Corrected marker
plt.title('a0 vs z')
plt.xlabel('z')
plt.ylabel('a0')
plt.xticks(np.linspace(min(z_data), max(z_data), 6))
plt.grid(True)

plt.tight_layout()
plt.show()

