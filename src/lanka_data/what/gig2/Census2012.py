import os

from lanka_data.what.FieldNameUtils import FieldNameUtils
from lanka_data.what.gig2.GIG2 import GIG2
from utils_future import Log

log = Log("Census2012")


class Census2012(GIG2):

    def __init__(self, title: str):
        super().__init__(
            title=title,
            region_group="regions",
            year="2012",
        )

    @classmethod
    def get_title_to_id_file_path(cls):
        return os.path.join(
            "src",
            "lanka_data",
            "what",
            "gig2",
            "census2012.datasets.json",
        )

    def get_source_info(self):
        description = self.get_title_to_description().get(self.title, "")
        return dict(
            source="Census of Population and Housing 2012",
            source_url="https://www.statistics.gov.lk"
            + "/Resource/en/Population"
            + "/CPH_2011/CPH_2012_5Per_Rpt.pdf",
            description=description,
        )

    @classmethod
    def get_what_to_whens(cls) -> dict[str, set[str]]:
        title_to_id = cls.get_title_to_id()
        what_to_whens = {}
        for title in title_to_id:
            what_to_whens[title] = ["2012"]
        return what_to_whens

    @classmethod
    def get_custom_data(cls, d):
        values = {}
        for k, v in d.items():
            if "total" in k:
                continue
            if "region_id" in k:
                continue
            values[FieldNameUtils.normalize(k)] = int(float(v))

        values = dict(sorted(values.items(), key=lambda item: -item[1]))
        total_value = sum(values.values())
        pct_values = {k: round(v / total_value, 4) for k, v in values.items()}

        return dict(
            values=values,
            total_value=total_value,
            pct_values=pct_values,
        )
