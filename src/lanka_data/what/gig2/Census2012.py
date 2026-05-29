import os

from lanka_data.what.gig2.GIG2 import GIG2
from utils_future import Log

log = Log("Census2012")


class Census2012(GIG2):

    def __init__(self, what_label: str):
        super().__init__(
            what_label=what_label,
            region_group="regions",
            year="2012",
        )

    @classmethod
    def get_what_label_to_id_file_path(cls):
        return os.path.join(
            "src", "lanka_data", "what", "gig2", "census2012.datasets.json"
        )

    @classmethod
    def get_source_info(cls):
        return dict(
            source="Census of Population and Housing 2012",
            source_url="https://www.statistics.gov.lk"
            + "/Resource/en/Population/CPH_2011/CPH_2012_5Per_Rpt.pdf",
        )

    @classmethod
    def get_custom_result(cls, d):
        values = {}
        for k, v in d.items():
            if "total" in k:
                continue
            if "entity_id" in k:
                continue
            values[k] = int(float(v))

        values = dict(sorted(values.items(), key=lambda item: -item[1]))
        total_value = sum(values.values())
        pct_values = {k: round(v / total_value, 4) for k, v in values.items()}

        return dict(
            values=values,
            total_value=total_value,
            pct_values=pct_values,
        )
