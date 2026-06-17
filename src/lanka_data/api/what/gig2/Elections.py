import os

from lanka_data.api.what.gig2.GIG2 import GIG2
from utils_future import Log

log = Log("Elections")


class Elections(GIG2):

    def __init__(self, title: str, when_label: str):
        if when_label == "Latest":
            if title == "Presidential":
                when_label = "2024"
            elif title == "Parliamentary":
                when_label = "2020"
            elif title == "Local":
                when_label = "2018"
            else:
                raise ValueError(
                    f"Unknown title: {title} for Latest when_label"
                )
        super().__init__(
            title=title,
            region_group="regions-ec",
            year=when_label,
        )

    def get_description(self):
        return f"Results of the {self.year} Sri Lankan {self.title} Election"

    @classmethod
    def get_title_to_id_file_path(cls):
        return os.path.join(
            "src",
            "lanka_data",
            "api",
            "what",
            "gig2",
            "elections.datasets.json",
        )

    @classmethod
    def get_source_info(cls):
        return dict(
            source="Election Commission of Sri Lanka",
            source_url="https://www.elections.gov.lk",
        )

    @classmethod
    def get_what_to_whens(cls) -> dict[str, set[str]]:
        return {
            "Presidential": [
                "1982",
                "1988",
                "1994",
                "1999",
                "2005",
                "2010",
                "2015",
                "2024",
            ],
            "Parliamentary": [
                "1989",
                "1994",
                "2000",
                "2001",
                "2004",
                "2010",
                "2015",
                "2020",
            ],
            "Local": [
                "2025",
            ],
        }

    @classmethod
    def extract_source_data_values(cls, d):

        electors = int(float(d["electors"]))
        polled = int(float(d["polled"]))
        valid = int(float(d["valid"]))
        rejected = int(float(d["rejected"]))

        summary = dict(
            electors=electors,
            polled=polled,
            valid=valid,
            rejected=rejected,
            p_turnout=round(polled / electors, 4),
            p_valid=round(valid / polled, 4),
            p_rejected=round(rejected / polled, 4),
        )
        votes_by_party = {}
        for k, v in d.items():
            if k in ["region_id", "electors", "polled", "valid", "rejected"]:
                continue
            votes_by_party[k] = int(float(v))

        votes_by_party = dict(
            sorted(votes_by_party.items(), key=lambda item: -item[1])
        )
        pct_votes_by_party = {
            k: round(v / valid, 4) for k, v in votes_by_party.items()
        }

        return dict(
            summary=summary,
            votes_by_party=votes_by_party,
            pct_votes_by_party=pct_votes_by_party,
            total_value=summary["polled"],
        )

    @classmethod
    def get_aggregated_value_data(cls, data_list):
        summary = {}
        for k in ["electors", "polled", "valid", "rejected"]:
            total = sum(d["summary"][k] for d in data_list)
            summary[k] = total
        summary["p_turnout"] = round(
            summary["polled"] / summary["electors"], 4
        )
        summary["p_valid"] = round(summary["valid"] / summary["polled"], 4)
        summary["p_rejected"] = round(
            summary["rejected"] / summary["polled"], 4
        )

        votes_by_party = {}
        for d in data_list:
            for k, v in d["votes_by_party"].items():
                votes_by_party[k] = votes_by_party.get(k, 0) + v
        votes_by_party = dict(
            sorted(votes_by_party.items(), key=lambda item: -item[1])
        )

        pct_votes_by_party = {
            k: round(v / summary["valid"], 4)
            for k, v in votes_by_party.items()
        }
        pct_votes_by_party = dict(
            sorted(pct_votes_by_party.items(), key=lambda item: -item[1])
        )

        return dict(
            summary=summary,
            votes_by_party=votes_by_party,
            pct_votes_by_party=pct_votes_by_party,
            total_value=summary["polled"],
            values=votes_by_party,
            pct_values=pct_votes_by_party,
        )

    @classmethod
    def get_values(cls, data):
        return data["votes_by_party"]

    @classmethod
    def get_pct_values(cls, data):
        return data["pct_votes_by_party"]
