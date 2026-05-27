import logging
import os

from lanka_data.what.gig2.GIG2 import GIG2

log = logging.getLogger(__name__)


class Elections(GIG2):

    @classmethod
    def get_what_label_to_id_file_path(cls):
        return os.path.join(
            "src", "lanka_data", "what", "gig2", "elections.datasets.json"
        )

    @classmethod
    def get_custom_results(cls, d):

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
        by_party = {}
        for k, v in d.items():
            if k in ["entity_id", "electors", "polled", "valid", "rejected"]:
                continue
            by_party[k] = int(float(v))

        by_party = dict(sorted(by_party.items(), key=lambda item: -item[1]))
        p_by_party = {k: round(v / valid, 4) for k, v in by_party.items()}

        return dict(
            summary=summary,
            by_party=by_party,
            p_by_party=p_by_party,
            source="Election Commission of Sri Lanka",
            source_url="https://www.elections.gov.lk",
        )
