# generate_data.py
# Run ONCE before any training. Saves p, theta, AND S so all scripts use identical data.

import numpy as np
from params import N, M, T, L, alpha, noise
from generator import generate_demand, generate_mobility

print("GENERATING DEMAND AND MOBILITY DATA")

np.random.seed(42)

# Generate content sizes
content_size = 10
S = np.random.uniform(0.5, 2.0, M) * content_size
print(f"\nContent sizes S = {np.round(S, 2)}")

print("\nGenerating demand...")
p = generate_demand(N, M, T, alpha, noise)
print(f"  Demand: {p.shape}")

print("Generating mobility...")
theta = generate_mobility(N, L, T, use_real=False)
print(f"  Mobility: {theta.shape}")

# Save ALL three arrays 
np.save("demand_data_p.npy",       p)
np.save("mobility_data_theta.npy", theta)
np.save("content_sizes_S.npy",     S)

print("\nSaved:")
print("  demand_data_p.npy")
print("  mobility_data_theta.npy")
print("  content_sizes_S.npy")
print("\nNow run:")
print("  python train.py")
print("  python train_dueling_ddqn.py")
print("  python compare.py")
