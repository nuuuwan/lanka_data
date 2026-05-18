from lanka_data.core.Query import Query
from lanka_data.data_repos.gig2 import GIG2


class ConsoleElectionMixin:

    def _is_election(self, path: str) -> bool:
        try:
            norm_key, _ = Query(path).gig2_key()
            return bool(
                norm_key and norm_key.startswith("governmentelections")
            )
        except Exception:  # noqa: BLE001
            return False

    def _split_election(self, result: dict) -> tuple[dict, dict]:
        cols = GIG2._SUMMARY_COLS  # lowercase frozenset
        first = next(iter(result.values()), None)
        if isinstance(first, dict):
            summary = {
                eid: {
                    k: v
                    for k, v in sub.items()
                    if k.lower() in cols
                }
                for eid, sub in result.items()
            }
            party = {
                eid: {
                    k: v
                    for k, v in sub.items()
                    if k.lower() not in cols
                }
                for eid, sub in result.items()
            }
        else:
            summary = {
                k: v
                for k, v in result.items()
                if k.lower() in cols
            }
            party = {
                k: v
                for k, v in result.items()
                if k.lower() not in cols
            }
        return summary, party
