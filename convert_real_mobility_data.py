# convert_real_mobility_data.py
# Convert real channel distance data to mobility model (theta)

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os

print("CONVERTING REAL CHANNEL DATA TO MOBILITY MODEL")

# Load data
excel_path = r"F:\7thsemreport\Duelingddqn28\Caching project\distance_snapshots.xlsx"
excel_file = pd.ExcelFile(excel_path)

# Get all sheets (10 time snapshots)
all_sheets = excel_file.sheet_names
print(f"\nFound {len(all_sheets)} time snapshots: {all_sheets}\n")

# Load first snapshot to understand structure
df_t1 = pd.read_excel(excel_path, sheet_name='Snapshot_t1')

N = 300  # Number of receivers (users)
L = 4    # Number of base stations (locations)
T_original = 10  # Original time snapshots
T_target = 24  # Target time slots

print(f"Detected Structure:")
print(f"  N = {N} users (Rx1-Rx300)")
print(f"  L = {L} locations (BS1, BS2, BS3, BS4)")
print(f"  T_original = {T_original} snapshots")
print(f"  T_target = {T_target} slots (interpolated)\n")

# Extract BS columns (last 4 columns are BS1, BS2, BS3, BS4)
bs_columns = ['BS1', 'BS2', 'BS3', 'BS4']

# Initialize theta array: theta[user, location, time]
theta = np.zeros((N, L, T_target))

print("Loading and processing all snapshots...")

# Load distance data from all sheets
distance_data = np.zeros((T_original, N, L))

for t_idx, sheet_name in enumerate(all_sheets):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    
    # Extract BS columns (distance to each base station)
    bs_distances = df[bs_columns].values  # Shape: (300, 4)
    distance_data[t_idx] = bs_distances
    
    print(f"  {sheet_name}: shape {bs_distances.shape}")

print("\n Loaded all distance data\n")

# Convert distances to probabilities
# Closer BS = higher probability
# Use inverse distance weighting: P(location) = (1/distance)^2 / sum((1/distance)^2)

print("Converting distances → probabilities...")

# Initialize theta with original time dimension
theta_original = np.zeros((N, L, T_original))

for t in range(T_original):
    for n in range(N):
        distances = distance_data[t, n, :]  # Distance to each BS
        
        # Avoid division by zero: add small epsilon
        distances = np.maximum(distances, 0.1)
        
        # Inverse distance weighting (closer = stronger signal)
        # Using power of 2 (common in signal propagation)
        weights = (1.0 / (distances ** 2))
        
        # Normalize to probability distribution
        theta_original[n, :, t] = weights / weights.sum()

print("Converted to probabilities\n")

# Interpolate from 10 snapshots to 24 time slots
print("Interpolating 10 snapshots → 24 time slots...")

# Original time points: 0, 1, 2, ..., 9
# Target time points: 0, 1, 2, ..., 23

t_original = np.arange(T_original)
t_target = np.linspace(0, T_original - 1, T_target)

# Now interpolate to target time dimension
theta = np.zeros((N, L, T_target))

for n in range(N):
    for l in range(L):
        # Interpolate for each user-location combination
        f = interp1d(t_original, theta_original[n, l, :], kind='cubic', fill_value='extrapolate')
        theta[n, l, :] = f(t_target)
        
        # Ensure probabilities stay in [0, 1]
        theta[n, l, :] = np.clip(theta[n, l, :], 0, 1)

# Renormalize so probabilities sum to 1 for each user at each time
for t in range(T_target):
    for n in range(N):
        sum_l = theta[n, :, t].sum()
        if sum_l > 0:
            theta[n, :, t] /= sum_l

# Final result: theta_interpolated
theta_interpolated = theta

print("Interpolated successfully\n")

# Save as .npy files
print("Saving mobility data...")

output_path_theta = "mobility_data_theta_real.npy"
np.save(output_path_theta, theta_interpolated)

print(f" Saved to: {output_path_theta}")
print(f"  Shape: {theta_interpolated.shape}")
print(f"  (N={N}, L={L}, T={T_target})\n")

# Generate statistics
print("STATISTICS OF CONVERTED DATA")

print(f"\nProbability ranges per location:")
for l in range(L):
    prob_array = theta_interpolated[:, l, :]
    print(f"  Location {l} (BS{l+1}):")
    print(f"    Min: {prob_array.min():.4f}")
    print(f"    Max: {prob_array.max():.4f}")
    print(f"    Mean: {prob_array.mean():.4f}")
    print(f"    Std: {prob_array.std():.4f}")

print(f"\nTime slot statistics:")
for t in range(0, T_target, 6):  # Show every 6th time slot
    total_prob = theta_interpolated[:, :, t].sum() / N
    print(f"  Time {t}: Probability sums to {total_prob:.4f} per user (should be ~1.0)")

print("\n Conversion complete!" )
