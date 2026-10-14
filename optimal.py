import numpy as np
import itertools

class OptimalCaching:

    def __init__(self, N, M, L, S, p, theta, T, r):
        self.N = N
        self.M = M
        self.L = L
        self.S = S
        self.p = p
        self.theta = theta
        self.T = T
        self.r = r

    # meeting probability 
    # def meet_prob(self, subset, k, t):
    #     prob = 0.0
    #     for l in range(self.L):
    #         prod = 1.0
    #         for i in subset:
    #             prod *= self.theta[i][l][t]
    #         prob += prod * self.theta[k][l][t]
    #     return prob

    def meet_prob(self, subset, k, t):
        prob = 0.0
        for l in range(self.L):
            not_meet = 1.0
            for i in subset:
                not_meet *= (1 - self.theta[i][l][t])

            meet_any = 1 - not_meet
            prob += meet_any * self.theta[k][l][t]

        return prob
    # subset score (S_k )
    def compute_subset_score(self, subset, m):
        subset = list(subset)
        val = 0.0

        for t in range(self.T):
            temp = 0.0

            # own demand
            for i in subset:
                temp += self.p[i][m][t]

            # meeting probability
            for k in range(self.N):
                if k not in subset:
                    meet_prob = self.meet_prob(subset, k, t)
                    temp += self.p[k][m][t] * meet_prob

            val += temp

        return val / self.T

    # MAIN OPTIMAL 
    def run(self):

        x_opt = np.zeros((self.N, self.M))
        users = list(range(self.N))

        for m in range(self.M):

            best_scores = [0.0] * (self.N + 1)
            best_subsets = [None] * (self.N + 1)

            # compute S_k for all k = 1..N
            for k in range(1, self.N + 1):

                best_score = -1
                best_subset = None

                for subset in itertools.combinations(users, k):
                    score = self.compute_subset_score(subset, m)

                    if score > best_score:
                        best_score = score
                        best_subset = subset

                best_scores[k] = best_score
                best_subsets[k] = best_subset

            # Decide optimal k using thresholds
            selected_subset = []

            for k in range(1, self.N + 1):
                gain = best_scores[k] - best_scores[k - 1]

                if self.r <= gain:
                    selected_subset = best_subsets[k]
                else:
                    break

            # Assign caching 
            if selected_subset is not None:
                for u in selected_subset:
                    x_opt[u][m] = self.S[m]

        return x_opt