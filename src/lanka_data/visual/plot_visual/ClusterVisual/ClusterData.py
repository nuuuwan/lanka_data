class ClusterData:
    MAX_ITERATIONS = 50

    @staticmethod
    def _init_centers(values, k):
        lo, hi = min(values), max(values)
        if hi <= lo or k < 2:
            return [lo]
        step = (hi - lo) / (k - 1)
        return [lo + i * step for i in range(k)]

    @staticmethod
    def _assign(values, centers):
        labels = []
        for value in values:
            distances = [abs(value - center) for center in centers]
            labels.append(distances.index(min(distances)))
        return labels

    @staticmethod
    def _update(values, labels, centers):
        new_centers = []
        for j, old_center in enumerate(centers):
            members = [v for v, lab in zip(values, labels) if lab == j]
            if members:
                new_centers.append(sum(members) / len(members))
            else:
                new_centers.append(old_center)
        return new_centers

    @classmethod
    def cluster(cls, values, k):
        values = [v for v in values if v is not None]
        if not values:
            return [], []
        k = max(1, min(k, len(set(values))))
        centers = cls._init_centers(values, k)
        labels = cls._assign(values, centers)
        for _ in range(cls.MAX_ITERATIONS):
            centers = cls._update(values, labels, centers)
            new_labels = cls._assign(values, centers)
            if new_labels == labels:
                break
            labels = new_labels
        return labels, centers
