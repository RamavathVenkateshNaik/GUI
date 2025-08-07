import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# List of data directories
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

# Extract directory names for legend labels
dir_labels = [os.path.basename(os.path.normpath(dir_path)) for dir_path in data_directories]

# Initialize storage for all data
all_data = {}

# Read data from all directories
print("Reading data from all directories...")
for i, data_dir in enumerate(data_directories):
    dataprep_path = os.path.join(data_dir, 'dataprep.csv')
    
    if os.path.exists(dataprep_path):
        try:
            df = pd.read_csv(dataprep_path)
            all_data[dir_labels[i]] = df
            print(f"✓ Loaded data from {dir_labels[i]}: {len(df)} rows")
        except Exception as e:
            print(f"✗ Error loading {dataprep_path}: {e}")
    else:
        print(f"✗ File not found: {dataprep_path}")

if not all_data:
    print("No data files found! Please check the directory paths.")
    exit()

# Create the plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Parameter Variations Across All Datasets', fontsize=16, fontweight='bold')

# Define colors for different datasets
colors = plt.cm.tab10(np.linspace(0, 1, len(all_data)))

# Plot 1: a1 vs z
ax1 = axes[0, 0]
for i, (label, df) in enumerate(all_data.items()):
    ax1.plot(df['z'], df['a1'], 'o-', color=colors[i], label=label, markersize=6, linewidth=2)
ax1.set_title('a1 vs z', fontsize=14, fontweight='bold')
ax1.set_xlabel('z', fontsize=12)
ax1.set_ylabel('a1', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot 2: n1 vs z
ax2 = axes[0, 1]
for i, (label, df) in enumerate(all_data.items()):
    ax2.plot(df['z'], df['n1'], 's-', color=colors[i], label=label, markersize=6, linewidth=2)
ax2.set_title('n1 vs z', fontsize=14, fontweight='bold')
ax2.set_xlabel('z', fontsize=12)
ax2.set_ylabel('n1', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot 3: n0 vs z
ax3 = axes[1, 0]
for i, (label, df) in enumerate(all_data.items()):
    ax3.plot(df['z'], df['n0'], 'd-', color=colors[i], label=label, markersize=6, linewidth=2)
ax3.set_title('n0 vs z', fontsize=14, fontweight='bold')
ax3.set_xlabel('z', fontsize=12)
ax3.set_ylabel('n0', fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot 4: a0 vs z
ax4 = axes[1, 1]
for i, (label, df) in enumerate(all_data.items()):
    ax4.plot(df['z'], df['a0'], 'x-', color=colors[i], label=label, markersize=8, linewidth=2)
ax4.set_title('a0 vs z', fontsize=14, fontweight='bold')
ax4.set_xlabel('z', fontsize=12)
ax4.set_ylabel('a0', fontsize=12)
ax4.grid(True, alpha=0.3)
ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout to prevent legend cutoff
plt.tight_layout()
plt.subplots_adjust(top=0.93, right=0.85)

# Save the combined plot
output_path = "/home/sujith/Documents/ML/n1a1data/17L/n1a1_parameters_all_datasets.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nCombined plot saved as: {output_path}")

# Show the plot
plt.show()
