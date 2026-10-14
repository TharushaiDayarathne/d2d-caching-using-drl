# train.py

import numpy as np
import torch
from datetime import datetime

from params import N, M, T, L, S, r, episodes, batch_size
from generator import compute_meeting_prob
from env import (compute_reactive_cost, compute_proactive_cost,
                 compute_demand_avg, compute_meeting_avg,
                 compute_marginal_score, get_state_marginal)
from ddqn import DDQNAgent

# Load data 
print("Loading pre-generated data...")
try:
    p     = np.load("demand_data_p.npy")
    theta = np.load("mobility_data_theta.npy")
    S_file = np.load("content_sizes_S.npy")
    # Override params S with the SAVED S so train/compare use identical values
    S[:] = S_file
    print(f"  Demand {p.shape}  Mobility {theta.shape}  S={np.round(S,2)}")
except FileNotFoundError:
    print("Data files not found!  Run:  python generate_data.py")
    exit(1)

# Precompute fixed quantities
print("\nPrecomputing statistics...")
demand_avg    = compute_demand_avg(p)                   # [N, M]
meeting_avg   = compute_meeting_avg(N, L, T, theta)     # [N]
reactive_cost = compute_reactive_cost(N, M, L, S, p, theta, T)

print(f"  Reactive cost  (C^R) : {reactive_cost:.4f}")
print(f"  Content sizes  S     : {np.round(S, 2)}")
print(f"  Caching cost   r     : {r}")

# Agent 
STATE_SIZE  = N + 7   # marginal_score + demand + meeting + x[:,m](N) + m + n + r + S_norm
ACTION_SIZE = 2       # 0=don't cache, 1=cache

agent = DDQNAgent(STATE_SIZE, ACTION_SIZE)
print(f"\nDDQN | state={STATE_SIZE} | actions={ACTION_SIZE} | episodes={episodes}\n")

# Training loop 
# Key design:
#   Each episode = N*M decisions (one per user-content pair, row-major order)
#   State uses DYNAMIC marginal score that decreases as redundancy increases
#   Reward = marginal-score-based shaped reward + normalized terminal gain
#   Best model saved whenever new best is found

SHAPING_ALPHA = 0.05

rewards_hist = []
best_gain    = -np.inf
best_x       = np.zeros((N, M))

for episode in range(episodes):
    x           = np.zeros((N, M))
    transitions = []

    for m in range(M):
        for n in range(N):
            state  = get_state_marginal(m, n, x, demand_avg, meeting_avg, S, r, M, N, L, T, p, theta)
            action = agent.select_action(state)

            if action == 1:
                x[n, m] = S[m]    # binary caching

            is_last = (m == M - 1) and (n == N - 1)

            # Shaped reward: uses MARGINAL score so diminishing returns are penalised
            already_caching_m = [k for k in range(N) if x[k, m] > 0 and k != n]
            marg  = compute_marginal_score(n, m, already_caching_m, N, L, T, p, theta)
            shaped = SHAPING_ALPHA * max(marg - r, 0.0) * (S[m] / S.max()) * float(action)

            if is_last:
                pc       = compute_proactive_cost(N, M, L, S, x, p, theta, T, r)
                gain     = reactive_cost - pc
                terminal = gain / (reactive_cost + 1e-8)
                reward   = shaped + terminal
                done     = True
            else:
                reward = shaped
                done   = False

            if not is_last:
                nxt_n      = (n + 1) % N
                nxt_m      = m + (1 if n == N - 1 else 0)
                next_state = get_state_marginal(nxt_m, nxt_n, x, demand_avg, meeting_avg,
                                                S, r, M, N, L, T, p, theta)
            else:
                next_state = np.zeros(STATE_SIZE, dtype=np.float32)

            transitions.append((state, action, reward, next_state, float(done)))

    for trans in transitions:
        agent.buffer.push(*trans)

    # Multiple gradient updates per episode
    ep_loss, updates = 0.0, 0
    n_updates = min(8, max(1, len(agent.buffer) // batch_size))
    for _ in range(n_updates):
        loss = agent.train_step()
        if loss is not None:
            ep_loss += loss
            updates += 1

    agent.decay_epsilon()
    if episode % 10 == 0:
        agent.update_target()

    # Metrics & best-model checkpoint
    pc_ep   = compute_proactive_cost(N, M, L, S, x, p, theta, T, r)
    gain_ep = reactive_cost - pc_ep
    avg_loss = ep_loss / max(updates, 1)
    rewards_hist.append(gain_ep)

    if gain_ep > best_gain:
        best_gain = gain_ep
        best_x    = x.copy()
        # Save best model immediately
        torch.save(agent.q_net.state_dict(), "ddqn_agent_best.pth")

    if episode % 25 == 0 or episode < 5:
        gain_pct = 100.0 * gain_ep / (reactive_cost + 1e-8)
        best_pct = 100.0 * best_gain / (reactive_cost + 1e-8)
        cached   = int(np.sum(x > 0))
        print(f"Ep {episode:4d} | Gain: {gain_ep:7.3f} ({gain_pct:5.1f}%) | "
              f"Best: {best_gain:.3f} ({best_pct:.1f}%) | "
              f"Loss: {avg_loss:.5f} | Cached: {cached}/{N*M} | eps: {agent.epsilon:.3f}")

# Save final model 
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
torch.save(agent.q_net.state_dict(), f"ddqn_agent_{timestamp}.pth")
torch.save(agent.q_net.state_dict(), "ddqn_agent_latest.pth")

print(f"  Reactive cost (C^R)     : {reactive_cost:.4f}")
print(f"  Best gain (saved model) : {best_gain:.4f}  ({100*best_gain/reactive_cost:.1f}%)")
print(f"  Avg gain (last 100 ep)  : {np.mean(rewards_hist[-100:]):.4f}")
print(f"  Best model -> ddqn_agent_best.pth    (use this in compare.py)")
print(f"  Last model -> ddqn_agent_latest.pth")
print(f"\n  Run: python compare.py")
