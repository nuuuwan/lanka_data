INF = float("inf")


class HungarianUtils:
    def __init__(self, cost):
        self.cost = cost
        self.n = len(cost)
        self.m = len(cost[0]) if cost else 0
        self.u = [0.0] * (self.n + 1)
        self.v = [0.0] * (self.m + 1)
        self.p = [0] * (self.m + 1)
        self.way = [0] * (self.m + 1)

    def _reduced(self, i0, j):
        return self.cost[i0 - 1][j - 1] - self.u[i0] - self.v[j]

    def _find_delta(self, used, minv, j0):
        i0 = self.p[j0]
        delta, j1 = INF, -1
        for j in range(1, self.m + 1):
            if used[j]:
                continue
            cur = self._reduced(i0, j)
            if cur < minv[j]:
                minv[j], self.way[j] = cur, j0
            if minv[j] < delta:
                delta, j1 = minv[j], j
        return delta, j1

    def _update_potentials(self, used, minv, delta):
        for j in range(0, self.m + 1):
            if used[j]:
                self.u[self.p[j]] += delta
                self.v[j] -= delta
            else:
                minv[j] -= delta

    def _augment(self, i):
        self.p[0] = i
        j0 = 0
        minv = [INF] * (self.m + 1)
        used = [False] * (self.m + 1)
        while True:
            used[j0] = True
            delta, j1 = self._find_delta(used, minv, j0)
            self._update_potentials(used, minv, delta)
            j0 = j1
            if self.p[j0] == 0:
                break
        self._relabel(j0)

    def _relabel(self, j0):
        while j0:
            j1 = self.way[j0]
            self.p[j0] = self.p[j1]
            j0 = j1

    def _solve(self):
        for i in range(1, self.n + 1):
            self._augment(i)
        result = [-1] * self.n
        for j in range(1, self.m + 1):
            if self.p[j] != 0:
                result[self.p[j] - 1] = j - 1
        return result

    @classmethod
    def solve(cls, cost):
        if not cost or not cost[0]:
            return []
        return cls(cost)._solve()
