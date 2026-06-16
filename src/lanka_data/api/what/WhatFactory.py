from lanka_data.api.what.BasicWhat import BasicWhat
from lanka_data.api.what.census2024.Census2024 import Census2024
from lanka_data.api.what.DiffWhat import DiffWhat
from lanka_data.api.what.gig2.Census2012 import Census2012
from lanka_data.api.what.gig2.Elections import Elections


class WhatFactory:
    @staticmethod
    def from_what_and_when(title: str, when_label: str):  # noqa: CFQ004
        if "-" in when_label:
            year1, year2 = when_label.split("-")
            what1 = WhatFactory.from_what_and_year(title, year1)
            what2 = WhatFactory.from_what_and_year(title, year2)
            return DiffWhat(what1, what2)

        return WhatFactory.from_what_and_year(title, when_label)

    @staticmethod
    def from_what_and_year(title: str, year: str):  # noqa: CFQ004
        if title == "Basic":
            return BasicWhat()

        if title in ["Parliamentary", "Presidential", "Local"]:
            return Elections(title, year)

        if year == "2012":
            return Census2012(title)

        if year == "2024" or year == "Latest":
            return Census2024(title)

        raise ValueError(f"Unknown title/year: {title}/{year}")

    @staticmethod
    def get_what_to_whens():
        idx = {}
        for cls in [BasicWhat, Elections, Census2012, Census2024]:
            for what, whens in cls.get_what_to_whens().items():
                if what not in idx:
                    idx[what] = []
                idx[what].extend(whens)

        idx = {what: sorted(list(whens)) for what, whens in idx.items()}
        idx = dict(sorted(idx.items(), key=lambda item: item[0]))
        return idx
