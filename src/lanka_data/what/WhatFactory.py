from lanka_data.what.BasicWhat import BasicWhat
from lanka_data.what.census2024.Census2024 import Census2024
from lanka_data.what.gig2.Census2012 import Census2012
from lanka_data.what.gig2.Elections import Elections


class WhatFactory:
    @classmethod
    def from_what_and_when(cls, title: str, when_label: str):  # noqa: CFQ004
        if title == "Basic":
            return BasicWhat()

        if "Election" in title:
            return Elections(title, when_label)

        if when_label == "2012":
            return Census2012(title)

        if when_label == "2024":
            return Census2024(title)

        raise ValueError(f"Unknown title: {title} or when_label: {when_label}")
