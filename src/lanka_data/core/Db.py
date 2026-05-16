from ..data_repos.gig2 import GIG2
from .Query import Query


class Db:
    def __call__(self, path: str) -> dict:
        return GIG2.query(Query(path))
