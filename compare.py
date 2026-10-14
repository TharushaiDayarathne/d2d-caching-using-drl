# compare.py  -  DDQN vs Dueling-DDQN vs Greedy vs Optimal 

import numpy as np
import torch
import os

from params import N, M, T, L, S, r
from env import (compute_reactive_cost, compute_proactive_cost,
                 compute_demand_avg, compute_meeting_avg,
                 compute_marginal_score, get_state_marginal)
from ddqn import DDQNAgent, DuelingDDQNAgent
from greedy import GreedyCachingR
from optimal import OptimalCaching

print("=" * 70)
print("  COMPARISON: DDQN vs Dueling-DDQN vs Greedy vs Optimal")
print("=" * 70)

# Load data
print("\nLoading data...")
try:
    p      = np.load("demand_data_p.npy")
    theta  = np.load("mobility_data_theta.npy")
    S_file = np.load("content_sizes_S.npy")
    S[:] = S_file    # override params.S with the saved value
    print(f"  Demand {p.shape}  Mobility {theta.shape}  S={np.round(S,2)}")
except FileNotFoundError:
    print("Data not found!  Run: python generate_data.py  then  python train.py")
    exit(1)

# Precompute
demand_avg    = compute_demand_avg(p)
meeting_avg   = compute_meeting_avg(N, L, T, theta)
reactive_cost = compute_reactive_cost(N, M, L, S, p, theta, T)

print(f"  Reactive cost (C^R) : {reactive_cost:.4f}")

STATE_SIZE  = N + 7
ACTION_SIZE = 2


def evaluate_rl_agent(agent, label):
    """Greedy (epsilon=0) rollout: N*M decisions, returns (x, proactive_cost, gain)."""
    agent.epsilon = 0.0
    x = np.zeros((N, M))
    for m in range(M):
        for n in range(N):
            state  = get_state_marginal(m, n, x, demand_avg, meeting_avg, S, r, M, N, L, T, p, theta)
            action = agent.select_action(state)
            if action == 1:
                x[n, m] = S[m]
    pc   = compute_proactive_cost(N, M, L, S, x, p, theta, T, r)
    gain = reactive_cost - pc
    print(f"  {label:<25}  cached {int(np.sum(x>0)):2d}/{N*M}  gain={gain:.4f}")
    return x, pc, gain


results = {}

# DDQN 
print("\n[1] Standard DDQN")
for path in ["ddqn_agent_best.pth", "ddqn_agent_latest.pth"]:
    if os.path.exists(path):
        ag = DDQNAgent(STATE_SIZE, ACTION_SIZE)
        ag.q_net.load_state_dict(torch.load(path, map_location="cpu"))
        tag = "DDQN (best)" if "best" in path else "DDQN (latest)"
        results[tag] = evaluate_rl_agent(ag, tag)
        break
else:
    print("  No DDQN model found.  Run: python train.py")

# Dueling DDQN
print("\n[2] Dueling DDQN")
for path in ["ddqn_dueling_best.pth", "ddqn_dueling_latest.pth"]:
    if os.path.exists(path):
        ag = DuelingDDQNAgent(STATE_SIZE, ACTION_SIZE)
        ag.q_net.load_state_dict(torch.load(path, map_location="cpu"))
        tag = "Dueling DDQN (best)" if "best" in path else "Dueling DDQN (latest)"
        results[tag] = evaluate_rl_agent(ag, tag)
        break
else:
    print("  No Dueling model found.  Run: python train_dueling_ddqn.py")

# Greedy
print("\n[3] Greedy")
x_greedy   = GreedyCachingR(N, M, L, S, p, theta, T, r).run()
pc_greedy  = compute_proactive_cost(N, M, L, S, x_greedy, p, theta, T, r)
g_greedy   = reactive_cost - pc_greedy
results["Greedy"] = (x_greedy, pc_greedy, g_greedy)
print(f"  {'Greedy':<25}  cached {int(np.sum(x_greedy>0)):2d}/{N*M}  gain={g_greedy:.4f}")

# Optimal
print("\n[4] Optimal")
if N <= 15:
    x_opt  = OptimalCaching(N, M, L, S, p, theta, T, r).run()
    pc_opt = compute_proactive_cost(N, M, L, S, x_opt, p, theta, T, r)
    g_opt  = reactive_cost - pc_opt
    results["Optimal"] = (x_opt, pc_opt, g_opt)
    print(f"  {'Optimal':<25}  cached {int(np.sum(x_opt>0)):2d}/{N*M}  gain={g_opt:.4f}")
else:
    print(f"  Skipped (N={N} > 15)")

# Summary table 
print(f"  {'Algorithm':<25} | {'Proactive Cost':>14} | {'Gain':>10} | {'Gain %':>7}")
print(f"  {'-'*25}-+-{'-'*14}-+-{'-'*10}-+-{'-'*7}")
print(f"  {'Reactive baseline':<25} | {reactive_cost:>14.4f} | {'---':>10} | {'---':>7}")
for name, (_, pc_, g_) in sorted(results.items(), key=lambda kv: -kv[1][2]):
    pct = 100.0 * g_ / (reactive_cost + 1e-8)
    print(f"  {name:<25} | {pc_:>14.4f} | {g_:>10.4f} | {pct:>6.1f}%")

# Gap to optimal
# if "Optimal" in results:
#     _, _, g_opt = results["Optimal"]
#     print(f"\n  Gap to Optimal  (positive = below optimal):")
#     for name, (_, _, g_) in results.items():
#         if name == "Optimal":
#             continue
#         gap = 100.0 * (g_opt - g_) / (g_opt + 1e-8)
#         print(f"    {name:<25}  {gap:+.1f}%")

# Caching matrices

print("  Caching matrices  (rows=users, cols=contents)")
for name, (x_, _, _) in results.items():
    print(f"\n  {name}:")
    print(np.round(x_, 2))

print("  Done.")
