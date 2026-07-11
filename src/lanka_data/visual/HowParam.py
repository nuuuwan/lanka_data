from dataclasses import dataclass


@dataclass(frozen=True)
class HowParam:
    name: str
    label: str
    description: str
    rank: int | None = None
    pct_rank: int | None = None
    needs_interval: bool = False

    def get_description(self) -> str:
        return self.description


HOW_PARAMS = {
    "1st": HowParam(
        name="1st",
        label="Most common",
        description="Highlights the most common category in each region",
        rank=0,
    ),
    "Top": HowParam(
        name="Top",
        label="Most common",
        description="Highlights the most common category in each region",
        rank=0,
    ),
    "2nd": HowParam(
        name="2nd",
        label="2nd most common",
        description="Highlights the 2nd most common category in each region",
        rank=1,
    ),
    "3rd": HowParam(
        name="3rd",
        label="3rd most common",
        description="Highlights the 3rd most common category in each region",
        rank=2,
    ),
    "Bottom": HowParam(
        name="Bottom",
        label="Least common",
        description="Highlights the least common category in each region",
        rank=-1,
    ),
    "1stPct": HowParam(
        name="1stPct",
        label="Most common share",
        description=(
            "Shows the percentage share of the most common category "
            "in each region"
        ),
        pct_rank=0,
    ),
    "2ndPct": HowParam(
        name="2ndPct",
        label="2nd most common share",
        description=(
            "Shows the percentage share of the 2nd most common category "
            "in each region"
        ),
        pct_rank=1,
    ),
    "3rdPct": HowParam(
        name="3rdPct",
        label="3rd most common share",
        description=(
            "Shows the percentage share of the 3rd most common category "
            "in each region"
        ),
        pct_rank=2,
    ),
    "Change": HowParam(
        name="Change",
        label="Change",
        description=(
            "Shows the change in the selected metric between two time periods. "
            "Requires an interval (two years) in the When field."
        ),
        needs_interval=True,
    ),
    "Top3": HowParam(
        name="Top3",
        label="Top 3 fields",
        description=(
            "Colors each region based on its top 3 categories combined, "
            "assigning a unique color to each unique combination"
        ),
    ),
    "Diversity": HowParam(
        name="Diversity",
        label="Diversity",
        description=(
            "Shows the Religious Diversity Index (RDI) for each region, "
            "measuring how evenly distributed the categories are"
        ),
    ),
    "DiversityPew": HowParam(
        name="DiversityPew",
        label="Pew diversity",
        description=(
            "Shows the Pew-adjusted Religious Diversity Index for each region, "
            "using grouped categories similar to Pew Research methodology"
        ),
    ),
}
