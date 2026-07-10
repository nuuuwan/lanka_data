from collections import deque


class ContiguityRepairMixin:
    @staticmethod
    def _to_label(centers, cells):
        pos = {
            (round(x, 9), round(y, 9)): i for i, (x, y) in enumerate(centers)
        }
        return {pos[(round(x, 9), round(y, 9))]: r for r, x, y in cells}

    @staticmethod
    def _to_cells(label, centers):
        return [[r, centers[i][0], centers[i][1]] for i, r in label.items()]

    @classmethod
    def _find_orphan(cls, label, adj):
        by_region = {}
        for i, r in label.items():
            by_region.setdefault(r, []).append(i)
        for r, indices in by_region.items():
            comps = cls._components(indices, adj)
            if len(comps) > 1:
                comps.sort(key=len)
                return r, set(comps[-1]), comps[0][0]
        return None

    @staticmethod
    def _trace(target, prev, core):
        path = [target]
        while path[-1] not in core:
            path.append(prev[path[-1]])
        path.reverse()
        return path

    @classmethod
    def _path_to_free(cls, core, free, adj):
        prev = {}
        seen = set(core)
        queue = deque(core)
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v in seen:
                    continue
                seen.add(v)
                prev[v] = u
                if v in free:
                    return cls._trace(v, prev, core)
                queue.append(v)
        return None

    @staticmethod
    def _shift(label, path, region):
        carry = region
        for idx in path[1:]:
            carry, label[idx] = label.get(idx), carry

    @classmethod
    def _repair_once(cls, label, n, adj):
        target = cls._find_orphan(label, adj)
        if target is None:
            return False
        region, core, orphan = target
        del label[orphan]
        free = set(range(n)) - set(label)
        path = cls._path_to_free(core, free, adj)
        if path is None:
            label[orphan] = region
            return False
        cls._shift(label, path, region)
        return True

    @classmethod
    def repair(cls, cells, centers):
        adj = cls._adjacency(centers)
        label = cls._to_label(centers, cells)
        n = len(centers)
        for _ in range(n * 4 + 100):
            if not cls._repair_once(label, n, adj):
                break
        return cls._to_cells(label, centers)
