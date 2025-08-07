import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Original y data provided (in float, scientific notation)
data = pd.read_csv('/home/sujith/Documents/ML/dataprep/37L/200K37L/xchdata.csv', header = None)

original_y = data[11].values
# Generate x values corresponding to the original data
#x = np.linspace(-1.099, 1.099, 17).reshape(-1, 1)
x = data[0].values.reshape(-1, 1)

# Given parameters
n0 = 0.986722
n1 = 0.464215
a0 = 0.923588
a1 = 1.189283
tsi = 4.887000
n_max = max(original_y)

# Compute predicted n(x)
pi_x_tsi = np.pi * x / tsi
cos2_term = np.cos(a0 * pi_x_tsi) ** 2
sinh2_term = np.sinh(a1 * pi_x_tsi) ** 2
predicted_y = (n0 * cos2_term + n1 * sinh2_term * cos2_term) * n_max

# Print x, original y, and predicted y
print(f"\n{'x':>8} {'Original y':>20} {'Predicted n(x)':>25}")
print("-" * 60)
for xi, yi, y_pred in zip(x, original_y, predicted_y):
    print(f"{xi.item():8.4f} {yi:20.4e} {y_pred.item():25.4e}")

# Plotting both
plt.plot(x, original_y, 'bo-', label='Original Data')
plt.plot(x, predicted_y, 'r--', label='Predicted n(x)')
plt.xlabel('x')
plt.ylabel('n(x)')
plt.title('Comparison of Original vs Predicted n(x)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

