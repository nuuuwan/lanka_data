from lanka_data.what.BasicWhat import BasicWhat
from lanka_data.what.census2024.Census2024 import Census2024
from lanka_data.what.gig2.Census2012 import Census2012
from lanka_data.what.gig2.Elections import Elections


class WhatFactory:
    @classmethod
    def from_what_and_when(  # noqa: CFQ004
        cls, what_label: str, when_label: str
    ):
        if what_label == "Basic":
            return BasicWhat()

        if "Election" in what_label:
            return Elections(what_label, when_label)

        if when_label == "2012":
            return Census2012(what_label)

        if when_label == "2024":
            return Census2024(what_label)

        raise ValueError(
            f"Unknown what_label: {what_label} or when_label: {when_label}"
        )
