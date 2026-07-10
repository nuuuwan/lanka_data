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
    def _update(vectors, labels, centers):
        new_centers = []
        for j, old_center in enumerate(centers):
            members = [v for v, lab in zip(vectors, labels) if lab == j]
            if not members:
                new_centers.append(old_center)
                continue
            dim = len(members[0])
            new_centers.append(
                [
                    sum(m[i] for m in members) / len(members)
                    for i in range(dim)
                ]
            )
        return new_centers

    @classmethod
    def cluster(cls, values, n_clusters):
        vectors = [cls._as_vector(v) for v in values if v is not None]
        if not vectors:
            return [], []
        n_clusters = max(1, min(n_clusters, len(set(vectors))))
        centers = cls._init_centers(vectors, n_clusters)
        labels = cls._assign(vectors, centers)
        for _ in range(cls.MAX_ITERATIONS):
            centers = cls._update(vectors, labels, centers)
            new_labels = cls._assign(vectors, centers)
            if new_labels == labels:
                break
            labels = new_labels
        return labels, centers
