# params.py

import numpy as np
import os

# System parameters 
N = 10
M = 5
L = 4
T = 24

# Demand model 
alpha = 0.8
noise = 0.05

content_size = 10
if os.path.exists("content_sizes_S.npy"):
    S = np.load("content_sizes_S.npy")
else:
    np.random.seed(0)
    S = np.random.uniform(0.5, 2.0, M) * content_size
    np.save("content_sizes_S.npy", S)

# Caching cost
r = 0.2

# RL hyperparameters 
episodes      = 500
batch_size    = 64
gamma         = 0.99
lr            = 3e-4

epsilon_start = 1.0
epsilon_end   = 0.05
epsilon_decay = 0.997
