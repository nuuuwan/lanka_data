import re


class QueryBase:

    _HOW_CANONICAL: dict[str, str] = {
        "json": "JSON",
        "bar": "Bar",
        "pie": "Pie",
        "map": "Map",
    }

    def __init__(self, path: str) -> None:
        parts = path.strip("/").split("/")
        if len(parts) == 3:
            self.where_raw, self.what_raw, self.when_raw = parts
            self.how_raw = "JSON"
            if self.when_raw.lower() in self._HOW_CANONICAL:
                raise ValueError(
                    f"{self.when_raw!r} looks like a <how> format,"
                    " not a <when> time period."
                    " Query must follow"
                    " /<where>/<what>/<when>[/<how>],"
                    f" e.g. add a time period before"
                    f" /{self.when_raw}."
                )
        elif len(parts) == 4:
            self.where_raw, self.what_raw, self.when_raw, self.how_raw = parts
            if self.how_raw.lower() not in self._HOW_CANONICAL:
                raise ValueError(
                    f"Unknown <how>: {self.how_raw!r}."
                    f" Must be one of: {
                        ', '.join(
                            self._HOW_CANONICAL.values())}."
                )
        else:
            raise ValueError(
                "Query must have 3 or 4 segments"
                f" (/<where>/<what>/<when>[/<how>]), got: {path!r}"
            )

    @property
    def how(self) -> str:
        return self._HOW_CANONICAL[self.how_raw.lower()]

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
