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

warnings.filterwarnings('ignore', category=UserWarning, message='.*convergence.*')
warnings.filterwarnings('ignore', category=RuntimeWarning)

layers = [17, 23, 31, 37, 53, 61, 69, 75]
temperatures = [15, 50, 77, 100, 150, 200, 250, 300]


data_directories  = [
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\17L\15K17L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\17L\50K17L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\17L\77K17L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\17L\100K17L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\17L\150K17L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\17L\200K17L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\17L\250K17L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\17L\300K17L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\23L\15K23L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\23L\50K23L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\23L\77K23L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\23L\100K23L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\23L\150K23L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\23L\200K23L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\23L\250K23L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\23L\300K23L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\31L\15K31L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\31L\50K31L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\31L\77K31L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\31L\100K31L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\31L\150K31L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\31L\200K31L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\31L\250K31L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\31L\300K31L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\37L\15K37L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\37L\50K37L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\37L\77K37L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\37L\100K37L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\37L\150K37L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\37L\200K37L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\37L\250K37L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\37L\300K37L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\53L\15K53L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\53L\50K53L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\53L\77K53L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\53L\100K53L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\53L\150K53L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\53L\200K53L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\53L\250K53L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\53L\300K53L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\61L\15K61L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\61L\50K61L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\61L\77K61L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\61L\100K61L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\61L\150K61L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\61L\200K61L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\61L\250K61L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\61L\300K61L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\69L\15K69L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\69L\50K69L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\69L\77K69L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\69L\100K69L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\69L\150K69L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\69L\200K69L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\69L\250K69L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\69L\300K69L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\75L\15K75L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\75L\50K75L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\75L\77K75L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\75L\100K75L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\75L\150K75L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\75L\200K75L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\75L\250K75L",
    r"C:\Users\PERSONAL\OneDrive\Desktop\GUI\n1a1data\75L\300K75L"  
]
for layer in layers:
    for temp in temperatures:
        data_directories.append(
            f"C:/Users/PERSONAL/OneDrive/Desktop/GUI/n1a1data/{layer}L/{temp}K{layer}L/"
        )

def calculate_tsi_and_x(num_layers):
    """Calculate Total Structural Interval (TSI) and x values for given layers"""
    tsi = (num_layers - 1) * 0.13575

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

    return tsi, np.array(x_values)

print("Loading data from all directories for all layers and temperatures...")

all_data = []
for directory in data_directories:

    try:
        basename = os.path.basename(os.path.normpath(directory))
        
        temp_str, layer_str = basename.split('K')
        temp = int(temp_str)
        layer = int(layer_str[:-1])  
    except Exception as e:
        print(f"Failed to parse temperature/layer from path '{directory}': {e}")
        continue

    csv_path = os.path.join(directory, "dataprep.csv")
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}, skipping...")
        continue

    try:
        df = pd.read_csv(csv_path)
        df['temperature'] = temp
        df['num_layers'] = layer
        tsi, _ = calculate_tsi_and_x(layer)
        df['tsi'] = tsi
        all_data.append(df)
        print(f"Loaded {layer}L - {temp}K: {len(df)} data points")
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")

if not all_data:
    print("No data loaded! Please check your file paths.")
    exit()

combined_df = pd.concat(all_data, ignore_index=True)
print(f"\nTotal combined data points: {len(combined_df)}")
print(f"Columns in dataset: {list(combined_df.columns)}")
print(f"Temperature range: {combined_df['temperature'].min()}K to {combined_df['temperature'].max()}K")
print(f"Layers in dataset: {sorted(combined_df['num_layers'].unique())}")
print(f"Voltage (z) range: {combined_df['z'].min():.4f} to {combined_df['z'].max():.4f}")


parameters = ['a1', 'n1', 'a0', 'n0']

X = combined_df[['z', 'temperature', 'tsi']].values

models = {}
results = {}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

print("\nTraining Gaussian Process Regression models...")

for i, param in enumerate(parameters):
    print(f"\n{'='*50}")
    print(f"Training GPR model for parameter: {param}")
    print(f"{'='*50}")

    y = combined_df[param].values
    print(f"{param} range: {y.min():.4f} to {y.max():.4f}")

    stratify_labels = combined_df['temperature'].astype(str) + "_" + combined_df['num_layers'].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_labels
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

def predict_parameter(param, voltage, temperature, num_layers):
    """Predict any parameter at given voltage, temperature, and layer count"""
    if param not in models:
        raise ValueError(f"Parameter {param} not found. Available: {list(models.keys())}")

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

def physical_model(x, a0, a1, n0, n1, tsi, n_max):
    """Physics-based prediction model"""
    pi_x_tsi = np.pi * x / tsi
    cos2_term = np.cos(a0 * pi_x_tsi) ** 2
    sinh2_term = np.sinh(a1 * pi_x_tsi) ** 2
    return (n0 * cos2_term + n1 * sinh2_term * cos2_term) * n_max

print(f"\n{'='*80}")
print("PHYSICAL MODEL PREDICTION (INTERACTIVE)")
print(f"{'='*80}")

def predict_physical_model():
    """Interactive prediction and comparison using physical model"""
    try:
        user_voltage = float(input("Enter voltage (z) value: "))
        user_temperature = float(input("Enter temperature (K): "))
        user_num_layers = int(input("Enter number of layers: "))

        closest_temp_idx = np.argmin(np.abs(np.array(temperatures) - user_temperature))
        closest_temp = temperatures[closest_temp_idx]
        closest_layer_idx = np.argmin(np.abs(np.array(layers) - user_num_layers))
        closest_layer = layers[closest_layer_idx]

        data_dir = f"C:/Users/PERSONAL/OneDrive/Desktop/GUI/n1a1data/{closest_layer}L/{closest_temp}K{closest_layer}L/"
        print(f"Using data from closest temperature-layer combo: {closest_temp}K - {closest_layer}L")
        print(f"Directory: {data_dir}")

        xchdata_path = os.path.join(data_dir, 'xchdata.csv')
        if not os.path.exists(xchdata_path):
            print(f"Error: '{xchdata_path}' not found!")
            return

        data = pd.read_csv(xchdata_path, header=None)
        x = data[0].values

        print(f"Available columns in xchdata.csv: {data.shape[1]} (0 to {data.shape[1]-1})")
        col_choice = int(input(f"Select column number for Y data (1 to {data.shape[1]-1}): "))
        if col_choice < 1 or col_choice >= data.shape[1]:
            print("Invalid column choice!")
            return

        original_y = data[col_choice].values
        n_max = max(original_y)
        print(f"Using column {col_choice}, n_max = {n_max:.4f}")

        predicted_params = {}
        for param in parameters:
            pred_val, uncertainty = predict_parameter(param, user_voltage, user_temperature, user_num_layers)
            predicted_params[param] = pred_val
            print(f"Predicted {param}: {pred_val:.6f} ± {uncertainty:.6f}")

        tsi, _ = calculate_tsi_and_x(user_num_layers)

        predicted_y = physical_model(
            x,
            predicted_params['a0'],
            predicted_params['a1'],
            predicted_params['n0'],
            predicted_params['n1'],
            tsi,
            n_max
        )

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(x, original_y, 'o-', label=f'Original Y[{col_choice}] at {closest_temp}K - {closest_layer}L', color='black', markersize=4)
        plt.plot(x, predicted_y, 's--', label=f'Predicted Y (User input: {user_temperature}K, {user_num_layers}L)', color='red', markersize=4)
        plt.title(f'Physical Model Prediction Comparison\nVoltage={user_voltage}, Temperature={user_temperature}K, Layers={user_num_layers}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        residuals = original_y - predicted_y
        plt.plot(x, residuals, 'o-', color='green', markersize=3)
        plt.axhline(0, color='red', linestyle='--', alpha=0.7)
        plt.title('Residuals (Original - Predicted)')
        plt.xlabel('x')
        plt.ylabel('Residual')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        mse = np.mean(residuals**2)
        mae = np.mean(np.abs(residuals))
        print(f"\nPrediction Statistics:")
        print(f"Mean Squared Error: {mse:.6f}")
        print(f"Mean Absolute Error: {mae:.6f}")
        print(f"Max Absolute Error: {np.max(np.abs(residuals)):.6f}")

    except ValueError as e:
        print(f"Invalid input: {e}")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"An error occurred: {e}")

print(f"\n{'='*80}")
print("✓ All models trained successfully!")
print("✓ Function `predict_parameter(param, voltage, temperature, num_layers)` available for predictions")
print("✓ Call `predict_physical_model()` for interactive physics-model based prediction and comparison")
print(f"{'='*80}")

print("\nRunning interactive physical model prediction...")
predict_physical_model()
