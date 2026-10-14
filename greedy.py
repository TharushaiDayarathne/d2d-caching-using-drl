import numpy as np

class GreedyCachingR:

    def __init__(self, N, M, L, S, p, theta, T, r):
        self.N = N
        self.M = M
        self.L = L
        self.S = S
        self.p = p
        self.theta = theta
        self.T = T
        self.r = r

    def compute_level1(self, m):
        scores = []
        for i in range(self.N):
            val = 0.0
            for t in range(self.T):
                temp = self.p[i][m][t]
                for j in range(self.N):
                    if j == i:
                        continue
                    meet = 0.0
                    for loc in range(self.L):
                        meet += self.theta[i][loc][t] * self.theta[j][loc][t]
                    temp += self.p[j][m][t] * meet
                val += temp
            val /= self.T
            scores.append((val, i))
        scores.sort(reverse=True)
        return scores

    def compute_next_level(self, selected, m):
        scores = []
        for j in range(self.N):
            if j in selected:
                continue
            val = 0.0
            for t in range(self.T):
                term1 = self.p[j][m][t]
                for k in selected:
                    meet = 0.0
                    for loc in range(self.L):
                        meet += self.theta[k][loc][t] * self.theta[j][loc][t]
                    term1 *= (1 - meet)

                term2 = 0.0
                for k in range(self.N):
                    if k == j:
                        continue
                    meet = 0.0
                    for loc in range(self.L):
                        prod = self.theta[j][loc][t] * self.theta[k][loc][t]
                        for s in selected:
                            prod *= (1 - self.theta[s][loc][t])
                        meet += prod
                    term2 += self.p[k][m][t] * meet

                val += term1 + term2
            val /= self.T
            scores.append((val, j))
        scores.sort(reverse=True)
        return scores

    def run(self):
        x = np.zeros((self.N, self.M))

        for m in range(self.M):
            selected = []

            scores = self.compute_level1(m)

            total_gain = 0.0

            for gain, i in scores:
                if gain <= self.r:
                    break

                x[i][m] = self.S[m]
                selected.append(i)
                total_gain += gain

                next_scores = self.compute_next_level(selected, m)

                for gain2, j in next_scores:
                    if gain2 <= self.r:
                        break

                    x[j][m] = self.S[m]
                    selected.append(j)
                    total_gain += gain2

                break

        return x
