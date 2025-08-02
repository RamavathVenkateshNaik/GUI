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
    "/home/sujith/Documents/ML/n1a1data/17L/15K17L/",
    "/home/sujith/Documents/ML/n1a1data/17L/50K17L/",
    "/home/sujith/Documents/ML/n1a1data/17L/77K17L/",
    "/home/sujith/Documents/ML/n1a1data/17L/100K17L/",
    "/home/sujith/Documents/ML/n1a1data/17L/150K17L/",
    "/home/sujith/Documents/ML/n1a1data/17L/200K17L/",
    "/home/sujith/Documents/ML/n1a1data/17L/250K17L/",
    "/home/sujith/Documents/ML/n1a1data/17L/300K17L/",
    "/home/sujith/Documents/ML/n1a1data/23L/15K23L/",
    "/home/sujith/Documents/ML/n1a1data/23L/50K23L/",
    "/home/sujith/Documents/ML/n1a1data/23L/77K23L/",
    "/home/sujith/Documents/ML/n1a1data/23L/100K23L/",
    "/home/sujith/Documents/ML/n1a1data/23L/150K23L/",
    "/home/sujith/Documents/ML/n1a1data/23L/200K23L/",
    "/home/sujith/Documents/ML/n1a1data/23L/250K23L/",
    "/home/sujith/Documents/ML/n1a1data/23L/300K23L/",
    "/home/sujith/Documents/ML/n1a1data/31L/15K31L/",
    "/home/sujith/Documents/ML/n1a1data/31L/50K31L/",
    "/home/sujith/Documents/ML/n1a1data/31L/77K31L/",
    "/home/sujith/Documents/ML/n1a1data/31L/100K31L/",
    "/home/sujith/Documents/ML/n1a1data/31L/150K31L/",
    "/home/sujith/Documents/ML/n1a1data/31L/200K31L/",
    "/home/sujith/Documents/ML/n1a1data/31L/250K31L/",
    "/home/sujith/Documents/ML/n1a1data/31L/300K31L/",
    "/home/sujith/Documents/ML/n1a1data/37L/15K37L/",
    "/home/sujith/Documents/ML/n1a1data/37L/50K37L/",
    "/home/sujith/Documents/ML/n1a1data/37L/77K37L/",
    "/home/sujith/Documents/ML/n1a1data/37L/100K37L/",
    "/home/sujith/Documents/ML/n1a1data/37L/150K37L/",
    "/home/sujith/Documents/ML/n1a1data/37L/200K37L/",
    "/home/sujith/Documents/ML/n1a1data/37L/250K37L/",
    "/home/sujith/Documents/ML/n1a1data/37L/300K37L/",
    "/home/sujith/Documents/ML/n1a1data/53L/15K53L/",
    "/home/sujith/Documents/ML/n1a1data/53L/50K53L/",
    "/home/sujith/Documents/ML/n1a1data/53L/77K53L/",
    "/home/sujith/Documents/ML/n1a1data/53L/100K53L/",
    "/home/sujith/Documents/ML/n1a1data/53L/150K53L/",
    "/home/sujith/Documents/ML/n1a1data/53L/200K53L/",
    "/home/sujith/Documents/ML/n1a1data/53L/250K53L/",
    "/home/sujith/Documents/ML/n1a1data/53L/300K53L/",
    "/home/sujith/Documents/ML/n1a1data/61L/15K61L/",
    "/home/sujith/Documents/ML/n1a1data/61L/50K61L/",
    "/home/sujith/Documents/ML/n1a1data/61L/77K61L/",
    "/home/sujith/Documents/ML/n1a1data/61L/100K61L/",
    "/home/sujith/Documents/ML/n1a1data/61L/150K61L/",
    "/home/sujith/Documents/ML/n1a1data/61L/200K61L/",
    "/home/sujith/Documents/ML/n1a1data/61L/250K61L/",
    "/home/sujith/Documents/ML/n1a1data/61L/300K61L/",
    "/home/sujith/Documents/ML/n1a1data/69L/15K69L/",
    "/home/sujith/Documents/ML/n1a1data/69L/50K69L/",
    "/home/sujith/Documents/ML/n1a1data/69L/77K69L/",
    "/home/sujith/Documents/ML/n1a1data/69L/100K69L/",
    "/home/sujith/Documents/ML/n1a1data/69L/150K69L/",
    "/home/sujith/Documents/ML/n1a1data/69L/200K69L/",
    "/home/sujith/Documents/ML/n1a1data/69L/250K69L/",
    "/home/sujith/Documents/ML/n1a1data/69L/300K69L/",
    "/home/sujith/Documents/ML/n1a1data/75L/15K75L/",
    "/home/sujith/Documents/ML/n1a1data/75L/50K75L/",
    "/home/sujith/Documents/ML/n1a1data/75L/77K75L/",
    "/home/sujith/Documents/ML/n1a1data/75L/100K75L/",
    "/home/sujith/Documents/ML/n1a1data/75L/150K75L/",
    "/home/sujith/Documents/ML/n1a1data/75L/200K75L/",
    "/home/sujith/Documents/ML/n1a1data/75L/250K75L/",
    "/home/sujith/Documents/ML/n1a1data/75L/300K75L/"  
]

# Temperature values and layer configurations
temperatures = [15, 50, 77, 100, 150, 200, 250, 300]
layers = [17, 23, 31, 37, 53, 61, 69, 75]

def calculate_tsi_and_x(num_layers):
    """Calculate TSI and X range for given number of layers"""
    tsi = (num_layers - 1) * 0.13575
    tsi = tsi * 1
    
    step = 0.13575
    half_points = num_layers // 2
    
    x_values = []
    
    if num_layers % 2 == 1:
        for i in range(half_points, 0, -1):
            x_values.append(-i * step)
        x_values.append(0.0)
        for i in range(1, half_points + 1):
            x_values.append(i * step)
    else:
        for i in range(half_points, 0, -1):
            x_values.append(-i * step + step/2)
        for i in range(half_points):
            x_values.append((i + 0.5) * step)
    
    x_values = np.array(x_values)
    return tsi, x_values

print("Loading data from all directories...")

# Load all data (excluding 75L for training)
all_data = []
for dir_idx, directory in enumerate(data_directories):
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} not found, skipping...")
        continue
        
    layer_idx = dir_idx // 8
    temp_idx = dir_idx % 8
    
    num_layers = layers[layer_idx]
    temperature = temperatures[temp_idx]
    
    # Skip 75L data for training
    if num_layers == 75:
        continue
    
    csv_path = os.path.join(directory, "dataprep.csv")
    
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df['temperature'] = temperature
            df['num_layers'] = num_layers
            
            tsi, _ = calculate_tsi_and_x(num_layers)
            df['tsi'] = tsi
            
            all_data.append(df)
            print(f"Loaded {num_layers}L-{temperature}K: {len(df)} data points")
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
    print(f"Layers: {sorted(combined_df['num_layers'].unique())}")
else:
    print("No data loaded! Please check your file paths.")
    exit()

# Parameters to model
parameters = ['a1', 'n1', 'a0', 'n0']

# Features: [voltage (z), temperature, tsi]
X = combined_df[['z', 'temperature', 'tsi']].values

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
    
    y = combined_df[param].values
    print(f"{param} range: {y.min():.4f} to {y.max():.4f}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    
    kernel = (C(1.0, (1e-3, 1e3)) * 
              RBF([1.0, 1.0, 1.0], (1e-2, 1e2)) + 
              WhiteKernel(1e-5, (1e-10, 1e-1)))
    
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        n_restarts_optimizer=20,
        random_state=42
    )
    
    print("Training model...")
    gpr.fit(X_train_scaled, y_train_scaled)
    
    y_pred_scaled, y_std_scaled = gpr.predict(X_test_scaled, return_std=True)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_std = y_std_scaled * scaler_y.scale_[0]
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Root Mean Square Error: {rmse:.4f}")
    print(f"Mean Prediction Uncertainty: {y_std.mean():.4f}")
    
    models[param] = {
        'gpr': gpr,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }
    
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
    
    ax = axes[i]
    scatter = ax.scatter(y_test, y_pred, alpha=0.7, c=X_test[:, 1], cmap='viridis', s=50)
    
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel(f'Actual {param}')
    ax.set_ylabel(f'Predicted {param}')
    ax.set_title(f'{param}: Actual vs Predicted (R²={r2:.3f})')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Temperature (K)')
    
plt.tight_layout()
plt.show()

print(f"\n{'='*80}")
print("SUMMARY OF ALL MODELS")
print(f"{'='*80}")
print(f"{'Parameter':<10} {'R²':<8} {'MAE':<8} {'RMSE':<8} {'Uncertainty':<12}")
print(f"{'-'*50}")
for param in parameters:
    r = results[param]
    print(f"{param:<10} {r['r2']:<8.4f} {r['mae']:<8.4f} {r['rmse']:<8.4f} {r['mean_uncertainty']:<12.4f}")

# Function to make predictions for any parameter
def predict_parameter(param, voltage, temperature, num_layers):
    """Predict any parameter at given voltage, temperature and number of layers"""
    if param not in models:
        raise ValueError(f"Parameter {param} not available. Choose from: {list(models.keys())}")
    
    tsi, _ = calculate_tsi_and_x(num_layers)
    
    model_data = models[param]
    gpr = model_data['gpr']
    scaler_X = model_data['scaler_X']
    scaler_y = model_data['scaler_y']
    
    X_new = np.array([[voltage, temperature, tsi]])
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

# Test 75L parameters prediction
print(f"\n{'='*80}")
print("TESTING 75L PARAMETERS PREDICTION")
print(f"{'='*80}")

for temp in temperatures:
    # Load 75L data for this temperature
    dir_idx = 5 * 8 + temperatures.index(temp)  # 75L is at index 5
    data_dir = data_directories[dir_idx]
    csv_path = os.path.join(data_dir, "dataprep.csv")
    
    if os.path.exists(csv_path):
        df_75L = pd.read_csv(csv_path)
        
        print(f"\n75L - {temp}K:")
        print(f"{'Parameter':<10} {'Predicted':<12} {'Original':<12} {'Difference':<12}")
        print(f"{'-'*48}")
        
        for param in parameters:
            # Use first voltage value for prediction
            voltage = df_75L['z'].iloc[20]
            predicted_val, _ = predict_parameter(param, voltage, temp, 75)
            original_val = df_75L[param].iloc[20]
            difference = abs(predicted_val - original_val)
            
            print(f"{param:<10} {predicted_val:<12.6f} {original_val:<12.6f} {difference:<12.6f}")

# Test specific example points
print(f"\n{'='*80}")
print("TESTING SPECIFIC EXAMPLE POINTS")
print(f"{'='*80}")

# Define test points
test_points = [
    {'voltage': 2.1925, 'temperature': 150, 'layers': 17},
    {'voltage': 2.4796, 'temperature': 150, 'layers': 17},
    {'voltage': 2.6710, 'temperature': 150, 'layers': 17},
    {'voltage': 2.1925, 'temperature': 150, 'layers': 23},
    {'voltage': 2.4796, 'temperature': 150, 'layers': 23},
    {'voltage': 2.6710, 'temperature': 150, 'layers': 23},
    {'voltage': 2.1925, 'temperature': 150, 'layers': 31},
    {'voltage': 2.4796, 'temperature': 150, 'layers': 31},
    {'voltage': 0.3742, 'temperature': 15, 'layers': 91},
    {'voltage': 0.5656, 'temperature': 100, 'layers': 83},
    {'voltage': 0.757, 'temperature': 150, 'layers': 83},
    {'voltage': 0.9484, 'temperature': 200, 'layers': 83}
]

for i, point in enumerate(test_points):
    voltage = point['voltage']
    temperature = point['temperature']
    layers_val = point['layers']
    
    print(f"\nTest Point {i+1}: V={voltage}, T={temperature}K, Layers={layers_val}")
    
    # Find closest data for comparison
    temp_diffs = [abs(t - temperature) for t in temperatures]
    closest_temp_idx = temp_diffs.index(min(temp_diffs))
    closest_temp = temperatures[closest_temp_idx]
    
    layer_diffs = [abs(l - layers_val) for l in layers]
    closest_layer_idx = layer_diffs.index(min(layer_diffs))
    closest_layer = layers[closest_layer_idx]
    
    dir_idx = closest_layer_idx * 8 + closest_temp_idx
    data_dir = data_directories[dir_idx]
    csv_path = os.path.join(data_dir, "dataprep.csv")
    
    if os.path.exists(csv_path):
        df_closest = pd.read_csv(csv_path)
        # Find closest voltage
        voltage_diffs = [abs(v - voltage) for v in df_closest['z']]
        closest_voltage_idx = voltage_diffs.index(min(voltage_diffs))
        
        print(f"Closest data: {closest_layer}L, {closest_temp}K, V={df_closest['z'].iloc[closest_voltage_idx]:.3f}")
        print(f"{'Parameter':<10} {'Predicted':<12} {'Original':<12} {'Difference':<12}")
        print(f"{'-'*48}")
        
        for param in parameters:
            predicted_val, _ = predict_parameter(param, voltage, temperature, layers_val)
            original_val = df_closest[param].iloc[closest_voltage_idx]
            difference = abs(predicted_val - original_val)
            
            print(f"{param:<10} {predicted_val:<12.6f} {original_val:<12.6f} {difference:<12.6f}")

# Example physical model prediction with plotting
def predict_and_plot_example():
    """Predict and plot an example case"""
    
    # Example parameters
    user_voltage = 0.8527
    user_temperature = 100
    user_num_layers = 75
    user_n_max = 1.13E+025
    
    print(f"\nExample Physical Model Prediction:")
    print(f"V={user_voltage}, T={user_temperature}K, Layers={user_num_layers}, n_max={user_n_max}")
    
    # Calculate TSI and x values for this configuration
    tsi, x_values = calculate_tsi_and_x(user_num_layers)
    print(f"Calculated TSI: {tsi:.6f}")
    print(f"X range: [{x_values.min():.6f}, {x_values.max():.6f}]")
    
    # Find closest data
    temp_diffs = [abs(t - user_temperature) for t in temperatures]
    closest_temp_idx = temp_diffs.index(min(temp_diffs))
    closest_temp = temperatures[closest_temp_idx]
    
    layer_diffs = [abs(l - user_num_layers) for l in layers]
    closest_layer_idx = layer_diffs.index(min(layer_diffs))
    closest_layer = layers[closest_layer_idx]
    
    dir_idx = closest_layer_idx * 8 + closest_temp_idx
    data_dir = data_directories[dir_idx]
    
    print(f"Using closest data: {closest_layer}L, {closest_temp}K")
    
    # Load xchdata.csv
    xchdata_path = os.path.join(data_dir, 'xchdata.csv')
    
    if os.path.exists(xchdata_path):
        data = pd.read_csv(xchdata_path, header=None)
        
        # Use column 1 for Y data
        col_choice = 9
        original_y = data[col_choice].values
        
        # Predict parameters
        predicted_params = {}
        for param in parameters:
            pred_value, uncertainty = predict_parameter(param, user_voltage, user_temperature, user_num_layers)
            predicted_params[param] = pred_value
            print(f"Predicted {param}: {pred_value:.6f} ± {uncertainty:.6f}")
        
        # Calculate physical model prediction using calculated x_values
        predicted_y = physical_model(
            x_values, 
            predicted_params['a0'], 
            predicted_params['a1'], 
            predicted_params['n0'], 
            predicted_params['n1'], 
            tsi, 
            user_n_max
        )
        
        # Plot comparison
        plt.figure(figsize=(12, 6))
        plt.plot(data[0].values, original_y, 'o-', label=f'Original Y[{col_choice}] ({closest_layer}L, {closest_temp}K)', color='black', markersize=4)
        plt.plot(x_values, predicted_y, 's--', label=f'Predicted Y ({user_num_layers}L, {user_temperature}K)', color='red', markersize=4)
        plt.title(f'Physical Model Prediction Comparison\nV={user_voltage}, T={user_temperature}K, Layers={user_num_layers}, n_max={user_n_max}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"\nPredicted Parameters Used:")
        for param in parameters:
            print(f"  {param}: {predicted_params[param]:.6f}")
        print(f"  TSI: {tsi:.6f} (calculated)")
        print(f"  n_max: {user_n_max}")

print(f"\n{'='*80}")
print("✓ All models trained successfully!")
print("✓ 75L parameters tested against original data")
print("✓ Example test points predicted and compared")
print(f"{'='*80}")

# Run example prediction and plotting
predict_and_plot_example()
