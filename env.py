# env.py  -  D2D Caching Environment

import numpy as np
import itertools

def compute_LtP(N, M, L, S, x, p, theta, t):
    LtP   = 0.0
    users = list(range(N))

    # Term 1 : SOME users meet (exactly subset a_n, size 2 to N-1)
    for m in range(M):
        for n in range(2, N):                       # subset SIZE n
            for subset in itertools.combinations(users, n):
                subset_set = set(subset)

                cache_sum = sum(x[k][m] for k in subset)
                remaining = S[m] - cache_sum
                if remaining <= 1e-9:
                    continue

                demand_sum = sum(p[k][m][t] for k in subset)

                # P(exactly this subset co-located at some l, all others NOT at l)
                meet_prob = 0.0
                for l in range(L):
                    prod_in = 1.0
                    for k in subset:
                        prod_in *= theta[k][l][t]
                    if prod_in < 1e-12:
                        continue
                    prod_out = 1.0
                    for j in users:
                        if j not in subset_set:
                            prod_out *= (1.0 - theta[j][l][t])
                    meet_prob += prod_in * prod_out

                LtP += remaining * demand_sum * meet_prob

    # Term 2 : ALL users meet
    for m in range(M):
        cache_sum = sum(x[n][m] for n in users)
        remaining = S[m] - cache_sum
        if remaining <= 1e-9:
            continue

        demand_sum = sum(p[n][m][t] for n in users)

        meet_prob = 0.0
        for l in range(L):
            prod = 1.0
            for n in users:
                prod *= theta[n][l][t]
            meet_prob += prod

        LtP += remaining * demand_sum * meet_prob

    # Term 3 : Each user ALONE)
    for m in range(M):
        for n in users:
            remaining = S[m] - x[n][m]   # constraint guarantees x[n][m] <= S[m]
            if remaining <= 1e-9:
                continue

            demand = p[n][m][t]

            alone_prob = 0.0
            for l in range(L):
                p_at_l = theta[n][l][t]
                if p_at_l < 1e-9:
                    continue
                others_not = 1.0
                for j in users:
                    if j != n:
                        others_not *= (1.0 - theta[j][l][t])
                alone_prob += p_at_l * others_not

            LtP += remaining * demand * alone_prob

    return max(LtP, 0.0)   # floating-point safety


# Cost helpers

def compute_reactive_cost(N, M, L, S, p, theta, T):
    """Time-averaged cost when NOTHING is cached (reactive baseline C^R)."""
    x_zero = np.zeros((N, M))
    total  = sum(compute_LtP(N, M, L, S, x_zero, p, theta, t) for t in range(T))
    return total / T


def compute_proactive_cost(N, M, L, S, x, p, theta, T, r):
    """Time-averaged cost + cache rent under caching plan x  (C^P)."""
    total_load = sum(compute_LtP(N, M, L, S, x, p, theta, t) for t in range(T))
    return total_load / T + r * float(np.sum(x))


# Pre-computable statistics

def compute_level1_scores(N, M, L, T, p, theta):
    """
    Level-1 score from Algorithm 1, Eq. (15):
      si[n,m] = (1/T) sum_t [ p[n][m][t]
                             + sum_{j!=n} p[j][m][t] * sum_l theta[n][l][t]*theta[j][l][t] ]

    Key insight: si[n,m] > r  means caching item m at user n is profitable.
    The greedy algorithm caches based on this threshold.
    """
    si = np.zeros((N, M), dtype=np.float64)
    for n in range(N):
        for m in range(M):
            val = 0.0
            for t in range(T):
                temp = p[n][m][t]
                for j in range(N):
                    if j == n:
                        continue
                    meet = sum(theta[n][l][t] * theta[j][l][t] for l in range(L))
                    temp += p[j][m][t] * meet
                val += temp
            si[n, m] = val / T
    return si


def compute_demand_avg(p):
    """Mean demand over time:  demand_avg[n, m] = (1/T) sum_t p[n][m][t]."""
    return p.mean(axis=2)   # shape [N, M]


def compute_meeting_avg(N, L, T, theta):
    """
    Mean meeting rate for each user:
      meeting_avg[n] = (1/(T*(N-1))) sum_t sum_{j!=n} sum_l theta[n][l][t]*theta[j][l][t]
    """
    avg   = np.zeros(N, dtype=np.float64)
    denom = T * max(N - 1, 1)
    for n in range(N):
        total = 0.0
        for t in range(T):
            for j in range(N):
                if j == n:
                    continue
                total += sum(theta[n][l][t] * theta[j][l][t] for l in range(L))
        avg[n] = total / denom
    return avg


# RL state builder

def get_state(m, n, x, si, demand_avg, meeting_avg, S, r, M, N):
    
    # Binary cache indicator (0 or 1) for each user for content m
    cache_indicators = (x[:, m] / S[m]).clip(0, 1) if S[m] > 0 else np.zeros(N)

    return np.array([
        si[n, m],
        demand_avg[n, m],
        meeting_avg[n],
        *cache_indicators,
        float(m) / max(M - 1, 1),
        float(n) / max(N - 1, 1),
        float(r),
        float(S[m]) / float(S.max()),
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Marginal score  (replaces static level-1 score in RL state/reward)
# ─────────────────────────────────────────────────────────────────────────────

def compute_marginal_score(n, m, already_caching_m, N, L, T, p, theta):
    """
    Marginal gain of caching item m at user n, given that users in
    'already_caching_m' already cache it.  This is the generalized
    greedy level-(k+1) score from Algorithm 2 of the paper.

    When already_caching_m is empty this reduces to the level-1 score si[n,m].

    Derivation (independence approximation for meeting events):
      Term 1:  p_n * P(n not already served)
               = p_n * prod_{k in S_prev}(1 - meet(k,n))
      Term 2:  for each other non-cached user j,
               p_j * meet(j,n) * prod_{k in S_prev}(1 - meet(k,j))
               = p_j * P(j meets n) * P(j not served by any already-cached user)

    This score DECREASES as more users are added to already_caching_m,
    capturing diminishing returns correctly.
    """
    val    = 0.0
    ac_set = set(already_caching_m)

    for t in range(T):
        # ── Term 1: user n's own demand, not already reachable ──────────────
        p_not_covered_n = 1.0
        for k in already_caching_m:
            meet_kn = sum(theta[k][l][t] * theta[n][l][t] for l in range(L))
            p_not_covered_n *= (1.0 - meet_kn)

        term1 = p[n][m][t] * p_not_covered_n

        # ── Term 2: other non-cached users served NEW-LY by n ───────────────
        term2 = 0.0
        for j in range(N):
            if j == n or j in ac_set:
                continue
            meet_jn = sum(theta[n][l][t] * theta[j][l][t] for l in range(L))
            if meet_jn < 1e-9:
                continue
            # P(j not already served by cached users)
            p_j_not_covered = 1.0
            for k in already_caching_m:
                meet_kj = sum(theta[k][l][t] * theta[j][l][t] for l in range(L))
                p_j_not_covered *= (1.0 - meet_kj)
            term2 += p[j][m][t] * meet_jn * p_j_not_covered

        val += term1 + term2

    return val / T


def get_state_marginal(m, n, x, demand_avg, meeting_avg, S, r, M, N, L, T, p, theta):
    """
    State vector using the DYNAMIC marginal score instead of static si[n,m].

    Feature                Size   Why it matters
    ──────────────────     ─────  ─────────────────────────────────────────
    marginal_score          1     Actual marginal gain (decreases with redundancy)
    demand_avg[n,m]         1     User n own average demand for item m
    meeting_avg[n]          1     How often n meets others (D2D sharing rate)
    x[:, m]  (0/1)          N     Who already caches item m
    m / (M-1)               1     Content index (normalized)
    n / (N-1)               1     User index    (normalized)
    r                       1     Caching cost threshold
    S[m] / S_max            1     Content size  (normalized)
    ──────────────────     ─────
    Total                  N+7   = 17 for N=10
    """
    already_caching_m = [k for k in range(N) if x[k, m] > 0 and k != n]
    marg = compute_marginal_score(n, m, already_caching_m, N, L, T, p, theta)

    cache_indicators = (x[:, m] / S[m]).clip(0, 1) if S[m] > 0 else np.zeros(N)

    return np.array([
        marg,
        demand_avg[n, m],
        meeting_avg[n],
        *cache_indicators,
        float(m) / max(M - 1, 1),
        float(n) / max(N - 1, 1),
        float(r),
        float(S[m]) / float(S.max()),
    ], dtype=np.float32)
