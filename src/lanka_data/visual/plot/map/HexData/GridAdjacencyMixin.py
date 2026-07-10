class GridAdjacencyMixin:
    NEIGHBOR_FACTOR2 = 1.1**2

    @staticmethod
    def _dist2(a, b):
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

    @classmethod
    def _neighbor_dist2(cls, centers):
        dists = []
        n = len(centers)
        for i in range(n):
            a = centers[i]
            for j in range(i + 1, n):
                d = cls._dist2(a, centers[j])
                if d > 0:
                    dists.append(d)
        return min(dists) if dists else None

    @classmethod
    def _link(cls, centers, thresh, adj):
        n = len(centers)
        for i in range(n):
            a = centers[i]
            for j in range(i + 1, n):
                if cls._dist2(a, centers[j]) <= thresh:
                    adj[i].add(j)
                    adj[j].add(i)

    @classmethod
    def _adjacency(cls, centers):
        adj = {i: set() for i in range(len(centers))}
        near = cls._neighbor_dist2(centers)
        if near is None:
            return adj
        cls._link(centers, near * cls.NEIGHBOR_FACTOR2, adj)
        return adj

    @classmethod
    def _flood(cls, root, remaining, adj):
        comp = []
        stack = [root]
        remaining.discard(root)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v in remaining:
                    remaining.discard(v)
                    stack.append(v)
        return comp

    @classmethod
    def _components(cls, indices, adj):
        remaining = set(indices)
        out = []
        while remaining:
            root = next(iter(remaining))
            out.append(cls._flood(root, remaining, adj))
        return out
