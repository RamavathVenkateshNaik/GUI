import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('/home/sujith/Documents/ML/dataprep/53L/200K53L/xchdata.csv', header=None)
potential = pd.read_csv('/home/sujith/Documents/ML/dataprep/53L/200K53L/phi_53L_1nm_all.csv', header=None)

# Get original data
original_y = data[21].values
original_phi = potential[24].values  # Assuming same column for potential
x_val = data[0].values.reshape(-1, 1)
x = 1e-009*x_val
# Given parameters for both equations
n0 = 0.08396
n1 = 0.1309
a0 = 2.136
a1 = 0.4694
tsi = 7.0602e-009
t_si = tsi  # Using same variable name as in your equation
n_max = 1.39E+20

# Additional parameters for equation 2 (you may need to define these)
q = 1.6e-19  # Elementary charge (C) - please adjust if different
eps_si = 1.04e-10  # Permittivity (F/m) - please adjust if different

# EQUATION 1: Compute predicted n(x)
pi_x_tsi = np.pi * x / tsi
cos2_term = np.cos(a0 * pi_x_tsi) ** 2
sinh2_term = np.sinh(a1 * pi_x_tsi) ** 2
predicted_y = (n0 * cos2_term + n1 * sinh2_term * cos2_term) * n_max

# EQUATION 2: Compute predicted phi_diff
# Precompute constants
pi = np.pi
a0pi_tsi = (a0 * pi) / t_si
a1pi_tsi = (a1 * pi) / t_si
x2 = x**2
x_tsi = x / t_si

# First part: n0-dependent term
term1 = (n0 / 2) * (
    (x2 / 2) +
    (1 - np.cos((2 * a0 * pi * x) / t_si)) / ((2 * a0pi_tsi) ** 2)
)

# Second part: n1-dependent terms
term2_1 = (np.cosh(2 * a1pi_tsi * x) - 1) / ((2 * a1pi_tsi)**2)
term2_2 = x2 / 2
term2_3 = (np.cos(2 * a0pi_tsi * x) - 1) / ((2 * a0pi_tsi)**2)
part1 = (term2_1 - term2_2 + term2_3)

# Third part: Cosine-Cosh mixing term
numerator = (a1pi_tsi)**2 - (a0pi_tsi)**2
denominator = ((a1pi_tsi)**2 + (a0pi_tsi)**2)**2
part2 = (1 / denominator) - (numerator/denominator) * np.cos(a1pi_tsi * x) * np.cosh(a0pi_tsi * x)

# Fourth part: Sin-Sinh mixing term
sin_term = np.sin(a1pi_tsi * x) * np.sinh(a0pi_tsi * x)
part3 = (2 * (a1pi_tsi) * (a0pi_tsi) / (denominator ** 2)) * sin_term

# Combine all parts
phic = np.median(original_phi)
predicted_phi = phic + (q / eps_si) * n_max * 1e6 * (term1 + ((n1/4) * (part1 + part2 + part3)))

# Create two separate graphs
plt.figure(figsize=(15, 6))

# Graph 1: Equation 1 - n(x) comparison
plt.subplot(1, 2, 1)
plt.plot(x, original_y, 'bo-', label='Original Data (xchdata.csv)')
plt.plot(x, predicted_y, 'r--', label='Predicted n(x) - Equation 1')
plt.xlabel('x')
plt.ylabel('n(x)')
plt.title('Equation 1: n(x) Comparison')
plt.legend()
plt.grid(True)

# Graph 2: Equation 2 - phi_diff comparison
plt.subplot(1, 2, 2)
plt.plot(x, original_phi, 'go-', label='Original Data (potential.csv)')
plt.plot(x, predicted_phi, 'm--', label='Predicted phi_diff - Equation 2')
plt.xlabel('x')
plt.ylabel('phi_diff')
plt.title('Equation 2: phi_diff Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
