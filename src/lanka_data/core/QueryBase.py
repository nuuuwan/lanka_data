import re


class QueryBase:

    def __init__(self, path: str) -> None:
        parts = path.strip("/").split("/")
        if len(parts) != 3:
            raise ValueError(
                "Query must have exactly 3 segments"
                f" (/<what>/<when>/<where>), got: {path!r}"
            )
        self.what_raw = parts[0]
        self.when_raw = parts[1]
        self.where_raw = parts[2]

    @staticmethod
    def normalize(s: str) -> str:
        """Lower-case with all separators (-, _, space) stripped."""
        return re.sub(r"[-_\s]", "", s).lower()

    @property
    def what_parts(self) -> list[str]:
        return self.what_raw.split(":") if self.what_raw != "*" else []

    @property
    def is_wildcard_what(self) -> bool:
        return self.what_raw == "*"

    @property
    def is_wildcard_when(self) -> bool:
        return self.when_raw == "*"

    @property
    def is_wildcard_where(self) -> bool:
        return self.where_raw == "*"

    @property
    def year(self) -> str | None:
        if self.is_wildcard_when:
            return None
        return self.when_raw[:4]
