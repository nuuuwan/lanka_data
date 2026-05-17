def _sort_by_val(d: dict) -> dict:
    return dict(
        sorted(
            d.items(),
            key=lambda kv: kv[1] if isinstance(kv[1], (int, float)) else -1,
            reverse=True,
        )
    )


def _pct_flat(d: dict) -> dict | None:
    nums = {
        k: v
        for k, v in d.items()
        if k != "Total" and isinstance(v, (int, float))
    }
    raw_total = d.get("Total")
    total = (
        raw_total
        if isinstance(raw_total, (int, float))
        else sum(nums.values())
    )
    if not nums or not total:
        return None
    out = {}
    for k, v in nums.items():
        pct = v / total * 100
        if pct >= 0.005:
            out[k] = round(pct, 2)
    return _sort_by_val(out) or None


def _entity_total(sub: dict):
    raw = sub.get("Total")
    if isinstance(raw, (int, float)):
        return raw
    nums = [
        v
        for k, v in sub.items()
        if k != "Total" and isinstance(v, (int, float))
    ]
    return sum(nums) if nums else None


class ConsoleFormatMixin:

    def _compute_p_values(self, result) -> dict | None:
        special = ("years", "entities", "measurements")
        skip = not isinstance(result, dict) or any(
            k in result for k in special
        )
        if skip:
            return None
        first = next(iter(result.values()), None)
        if isinstance(first, dict):
            out = {k: _pct_flat(v) for k, v in result.items()}
            out = {k: v for k, v in out.items() if v is not None}
        else:
            out = _pct_flat(result)
        return out or None

    def _total_and_strip(self, result) -> tuple:
        special = ("years", "entities", "measurements")
        if not isinstance(result, dict) or any(k in result for k in special):
            return result, None, None
        first = next(iter(result.values()), None)
        if isinstance(first, dict):
            total_value = {
                eid: _entity_total(sub) for eid, sub in result.items()
            }
            values = {
                eid: _sort_by_val(
                    {k: v for k, v in sub.items() if k != "Total"}
                )
                for eid, sub in result.items()
            }
            return values, total_value, len(next(iter(values.values()), {}))
        raw = result.get("Total")
        nums = [
            v
            for k, v in result.items()
            if k != "Total" and isinstance(v, (int, float))
        ]
        if isinstance(raw, (int, float)):
            total_v = raw
        elif nums:
            total_v = sum(nums)
        else:
            total_v = None
        values = _sort_by_val(
            {k: v for k, v in result.items() if k != "Total"}
        )
        return values, total_v, len(values)
