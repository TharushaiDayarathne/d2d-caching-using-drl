# generator.py

import numpy as np


def generate_demand(N, M, T, alpha=0.8, noise=0.05):
    ranks = np.arange(1, M + 1)
    base_pop = 1 / (ranks ** alpha)
    base_pop /= base_pop.sum()

    p = np.zeros((N, M, T))

    for t in range(T):
        time_factor = 0.5 + 0.5 * np.sin((t / T) * 2 * np.pi)

        time_factor = max(time_factor, 1e-3)

        for n in range(N):
            user_pref = base_pop + noise * np.random.rand(M)
            user_pref /= user_pref.sum()

            p[n, :, t] = user_pref * time_factor

        sums = p[:, :, t].sum(axis=1, keepdims=True)

        # Avoid division by zero
        sums[sums == 0] = 1e-8

        p[:, :, t] /= sums

    return p


def generate_mobility(N, L, T, use_real=False, noise=0.05):
    """
    Generate or load mobility data.
    
    Args:
        N: Number of users
        L: Number of locations
        T: Number of time slots
        use_real: If True, load real mobility data (mobility_data_theta_real.npy)
                 If False, generate synthetic mobility data
        noise: Noise level for Markov transitions (synthetic only)
    
    Returns:
        theta[N, L, T]: Mobility probability distribution
    """
    
    # Try to load REAL mobility data if requested
    if use_real:
        import os
        if os.path.exists('mobility_data_theta_real.npy'):
            try:
                theta_real_full = np.load('mobility_data_theta_real.npy')
                print(f"OK Loaded REAL mobility data: shape {theta_real_full.shape}")
                
                # Extract first N users if needed
                if N < theta_real_full.shape[0]:
                    theta_real = theta_real_full[:N, :L, :T]
                    print(f"   Extracted first {N} users (out of {theta_real_full.shape[0]})")
                    print(f"   Result shape: {theta_real.shape}")
                    return theta_real
                else:
                    theta_real = theta_real_full[:N, :L, :T]
                    print(f"   Using shape: {theta_real.shape}")
                    return theta_real
            except Exception as e:
                print(f"WARNING Could not load real mobility data: {e}")
                print(f"   Falling back to synthetic generation...")
        else:
            print(f"WARNING mobility_data_theta_real.npy not found")
            print(f"   Falling back to synthetic generation...")
    
    # Generate synthetic mobility with dynamic L
    theta = np.zeros((N, L, T))

    for n in range(N):
        # Initialize with random distribution over L locations
        current = np.ones(L) / L
        
        for t in range(T):
            # Create transition matrix dynamically for current L
            # Simple Markov chain: stay at current location with bias, or move randomly
            
            if t < 8:  # Morning: mostly at home (location 0)
                P = np.eye(L) * 0.8
                P[0, :] += 0.2 / L
            elif t < 16:  # Afternoon: move around more
                P = np.ones((L, L)) / L
                np.fill_diagonal(P, 0.7)
            else:  # Evening: return home
                P = np.eye(L) * 0.9
                P[0, :] += 0.1 / L
            
            # Ensure rows sum to 1
            P = P / P.sum(axis=1, keepdims=True)
            
            # Apply transition
            current = current @ P
            current += noise * np.random.rand(L)
            current = np.maximum(current, 0)  # Ensure non-negative
            current /= current.sum()  # Normalize to probability
            
            theta[n, :, t] = current

    return theta


def compute_meeting_prob(theta):
    N, L, T = theta.shape
    meet = np.zeros((N, N, T))

    for t in range(T):
        for i in range(N):
            for j in range(N):
                meet[i, j, t] = np.sum(theta[i, :, t] * theta[j, :, t])

    return meet