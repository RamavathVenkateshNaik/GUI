import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
import warnings

# Suppress convergence warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*convergence.*')
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Your data directories
data_directories = [
    "/home/sujith/Documents/ML/dataprep/17L/15K17L/",
    "/home/sujith/Documents/ML/dataprep/17L/50K17L/",
    "/home/sujith/Documents/ML/dataprep/17L/77K17L/",
    "/home/sujith/Documents/ML/dataprep/17L/100K17L/",
    "/home/sujith/Documents/ML/dataprep/17L/150K17L/",
    "/home/sujith/Documents/ML/dataprep/17L/200K17L/",
    "/home/sujith/Documents/ML/dataprep/17L/250K17L/",
    "/home/sujith/Documents/ML/dataprep/17L/300K17L/"
]

# Temperature values corresponding to each directory
temperatures = [15, 50, 77, 100, 150, 200, 250, 300]

print("Loading data from all temperature directories...")

# Load all data
all_data = []
for directory, temp in zip(data_directories, temperatures):
    csv_path = os.path.join(directory, "dataprep.csv")
    
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df['temperature'] = temp
            all_data.append(df)
            print(f"Loaded {temp}K: {len(df)} data points")
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")
    else:
        print(f"File not found: {csv_path}")

# Combine all data
if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal combined data points: {len(combined_df)}")
    print(f"Columns: {list(combined_df.columns)}")
    print(f"Temperature range: {combined_df['temperature'].min()}K to {combined_df['temperature'].max()}K")
    print(f"Voltage (z) range: {combined_df['z'].min():.4f} to {combined_df['z'].max():.4f}")
else:
    print("No data loaded! Please check your file paths.")
    exit()

# Parameters to model
parameters = ['a1', 'n1', 'a0', 'n0']

# Features: [voltage (z), temperature]
X = combined_df[['z', 'temperature']].values

# Dictionary to store models and results
models = {}
results = {}

# Create a 2x2 plot for actual vs predicted
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

print("\nTraining Gaussian Process Regression models...")

for i, param in enumerate(parameters):
    print(f"\n{'='*50}")
    print(f"Training GPR model for parameter: {param}")
    print(f"{'='*50}")
    
    # Target variable
    y = combined_df[param].values
    print(f"{param} range: {y.min():.4f} to {y.max():.4f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=combined_df['temperature']
    )
    
    # Scale features and target
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    
    # Define kernel
    kernel = (C(1.0, (1e-3, 1e3)) * 
              RBF([1.0, 1.0], (1e-2, 1e2)) + 
              WhiteKernel(1e-5, (1e-10, 1e-1)))
    
    # Create and train GPR
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        n_restarts_optimizer=15,
        random_state=42
    )
    
    print("Training model...")
    gpr.fit(X_train_scaled, y_train_scaled)
    
    # Make predictions
    y_pred_scaled, y_std_scaled = gpr.predict(X_test_scaled, return_std=True)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_std = y_std_scaled * scaler_y.scale_[0]
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Root Mean Square Error: {rmse:.4f}")
    print(f"Mean Prediction Uncertainty: {y_std.mean():.4f}")
    
    # Store model and scalers
    models[param] = {
        'gpr': gpr,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }
    
    # Store results
    results[param] = {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'mean_uncertainty': y_std.mean(),
        'y_test': y_test,
        'y_pred': y_pred,
        'y_std': y_std,
        'X_test': X_test
    }
    
    # Plot Actual vs Predicted
    ax = axes[i]
    scatter = ax.scatter(y_test, y_pred, alpha=0.7, c=X_test[:, 1], cmap='viridis', s=50)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Labels and title
    ax.set_xlabel(f'Actual {param}')
    ax.set_ylabel(f'Predicted {param}')
    ax.set_title(f'{param}: Actual vs Predicted (R²={r2:.3f})')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add colorbar for temperature
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Temperature (K)')
    
plt.tight_layout()
plt.show()

# Summary table
print(f"\n{'='*80}")
print("SUMMARY OF ALL MODELS")
print(f"{'='*80}")
print(f"{'Parameter':<10} {'R²':<8} {'MAE':<8} {'RMSE':<8} {'Uncertainty':<12}")
print(f"{'-'*50}")
for param in parameters:
    r = results[param]
    print(f"{param:<10} {r['r2']:<8.4f} {r['mae']:<8.4f} {r['rmse']:<8.4f} {r['mean_uncertainty']:<12.4f}")

# Function to make predictions for any parameter
def predict_parameter(param, voltage, temperature):
    """Predict any parameter at given voltage and temperature"""
    if param not in models:
        raise ValueError(f"Parameter {param} not available. Choose from: {list(models.keys())}")
    
    model_data = models[param]
    gpr = model_data['gpr']
    scaler_X = model_data['scaler_X']
    scaler_y = model_data['scaler_y']
    
    X_new = np.array([[voltage, temperature]])
    X_new_scaled = scaler_X.transform(X_new)
    y_pred_scaled, y_std_scaled = gpr.predict(X_new_scaled, return_std=True)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_std = y_std_scaled * scaler_y.scale_[0]
    return y_pred[0], y_std[0]

# Physics-Based Model Function
def physical_model(x, a0, a1, n0, n1, tsi, n_max):
    """Physics-based model equation"""
    pi_x_tsi = np.pi * x / tsi
    cos2_term = np.cos(a0 * pi_x_tsi) ** 2
    sinh2_term = np.sinh(a1 * pi_x_tsi) ** 2
    return (n0 * cos2_term + n1 * sinh2_term * cos2_term) * n_max

# Main function for physical model prediction and comparison
print(f"\n{'='*80}")
print("STARTING PHYSICAL MODEL PREDICTION")
print(f"{'='*80}")

# Get user input
user_voltage = float(input("Enter voltage (z) value: "))
user_temperature = float(input("Enter temperature (K): "))
n_max = float(input("Enter n_max value: "))

# Predict parameters using GPR models
predicted_params = {}
print(f"\nPredicted parameters for V={user_voltage}, T={user_temperature}K:")
for param in parameters:
    pred_value, uncertainty = predict_parameter(param, user_voltage, user_temperature)
    predicted_params[param] = pred_value
    print(f"{param}: {pred_value:.6f} ± {uncertainty:.6f}")

# Constants
tsi = 2.17

# Load data for x values from xchdata.csv
xch_data = pd.read_csv('xchdata.csv', header=None)
x = xch_data[0].values  # Use column [0] for x values

# Load data for comparison from all_charge.csv
all_charge_path = "/home/sujith/Documents/ML/dataprep/17L/all_charge.csv"
if not os.path.exists(all_charge_path):
    print(f"Error: {all_charge_path} not found!")
    exit()

data = pd.read_csv(all_charge_path, header=None)

print(f"Available columns in all_charge.csv: {data.shape[1]} columns (0 to {data.shape[1]-1})")
col_choice = int(input(f"Enter column number for original y data (0 to {data.shape[1]-1}): "))

if col_choice < 0 or col_choice >= data.shape[1]:
    print("Invalid column choice!")
    exit()

original_y = data.iloc[:, col_choice].values
original_y = original_y[~pd.isna(original_y)]  # Remove NaN values

# Ensure x and original_y have the same length
min_length = min(len(x), len(original_y))
x = x[:min_length]
original_y = original_y[:min_length]

# Calculate physical model prediction using GPR predicted parameters
predicted_y = physical_model(
    x, 
    predicted_params['a0'], 
    predicted_params['a1'], 
    predicted_params['n0'], 
    predicted_params['n1'], 
    tsi, 
    n_max
)

# Plot comparison
plt.figure(figsize=(12, 6))
plt.plot(x, original_y, 'o-', label=f'Original Y[{col_choice}]', color='black', markersize=4)
plt.plot(x, predicted_y, 's--', label=f'Predicted Y using GPR params at {user_temperature}K', color='red', markersize=4)
plt.title(f'Physical Model Prediction Comparison\nV={user_voltage}, T={user_temperature}K, n_max={n_max}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"\n{'='*60}")
print("PREDICTION COMPLETE")
print(f"{'='*60}")
