class ClusterData:
    MAX_ITERATIONS = 50

    @staticmethod
    def _as_vector(value):
        if isinstance(value, (int, float)):
            return (float(value),)
        return tuple(float(x) for x in value)

    @staticmethod
    def _distance(a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    @staticmethod
    def _init_centers(vectors, n_clusters):
        unique = sorted(set(vectors))
        if len(unique) < 2 or n_clusters < 2:
            return [list(unique[0])]
        step = (len(unique) - 1) / (n_clusters - 1)
        return [list(unique[round(i * step)]) for i in range(n_clusters)]

    @classmethod
    def _assign(cls, vectors, centers):
        labels = []
        for vec in vectors:
            distances = [cls._distance(vec, center) for center in centers]
            labels.append(distances.index(min(distances)))
        return labels

    @staticmethod
    def _weighted_center(members, weights):
        total = sum(weights)
        if total <= 0:
            total = len(members)
            weights = [1.0] * len(members)
        dim = len(members[0])
        return [
            sum(w * m[i] for m, w in zip(members, weights)) / total
            for i in range(dim)
        ]

    @classmethod
    def _update(cls, vectors, labels, centers, weights):
        new_centers = []
        for j, old_center in enumerate(centers):
            members, member_weights = [], []
            for vec, lab, weight in zip(vectors, labels, weights):
                if lab == j:
                    members.append(vec)
                    member_weights.append(weight)
            if not members:
                new_centers.append(old_center)
                continue
            new_centers.append(cls._weighted_center(members, member_weights))
        return new_centers

    @staticmethod
    def _as_weights(weights, n):
        if weights is None:
            return [1.0] * n
        return [float(w) for w in weights]

    @classmethod
    def cluster(cls, values, n_clusters, weights=None):
        keep = [
            (v, w)
            for v, w in zip(values, cls._as_weights(weights, len(values)))
            if v is not None
        ]
        vectors = [cls._as_vector(v) for v, _ in keep]
        weights = [w for _, w in keep]
        if not vectors:
            return [], []
        n_clusters = max(1, min(n_clusters, len(set(vectors))))
        centers = cls._init_centers(vectors, n_clusters)
        labels = cls._assign(vectors, centers)
        for _ in range(cls.MAX_ITERATIONS):
            centers = cls._update(vectors, labels, centers, weights)
            new_labels = cls._assign(vectors, centers)
            if new_labels == labels:
                break
            labels = new_labels
        return labels, centers
