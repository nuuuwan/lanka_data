import os
from functools import cache

from lanka_data.what.What import What
from utils_future import WWW, JSONFile, Log

log = Log("Census2024")


class Census2024(What):

    def __init__(self, title: str):
        super().__init__(title)

    @classmethod
    def metadata_file_path(cls) -> str:
        return os.path.join(
            "src",
            "lanka_data",
            "what",
            "census2024",
            "census2024.metadata.json",
        )

    @classmethod
    def get_metadata(cls) -> dict:
        return JSONFile(cls.metadata_file_path()).read()

    @classmethod
    @cache
    def get_title_to_path(cls) -> dict:
        metadata = cls.get_metadata()
        label_to_path = {}
        for level1 in metadata:
            for level2, label in metadata[level1].items():
                if label in label_to_path:
                    continue
                path = f"{level1}/{level2}"
                label_to_path[label] = path
        return label_to_path

    @classmethod
    def clean(cls, d):
        values = {}
        for k, v in d.items():
            if "region" in k:
                continue
            if "total" in k:
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

    @classmethod
    def get_source_info(cls):
        return dict(
            source="Census of Population and Housing 2024",
            source_url="https://www.statistics.gov.lk"
            + "/Population/StaticalInformation/CPH2024",
        )

    def get_base_data_list(self) -> list[dict]:
        label_to_path = self.get_title_to_path()
        path = label_to_path.get(self.title)
        url = (
            "https://raw.githubusercontent.com"
            + "/nuuuwan/lk_census_2024/refs/heads/main"
            + f"/data/{path}/data.json"
        )
        raw_data_list = WWW(url).read_json()
        return raw_data_list

    def get_base_data_region_year(self) -> str:
        return "2024"

    @classmethod
    def get_where_to_what_id_map(cls, regions) -> dict:
        idx = {}
        for region in regions.raw_region_data_list:
            region_id = region["id"]
            current_ids = region.get("current_ids", [region_id])
            idx[region_id] = current_ids
        return idx

    @classmethod
    def _get_mapped_data_for_region(
        cls, region_id, current_ids, raw_data_idx, raw_region_data_idx
    ):
        raw_data_list_for_region = []
        for current_id in current_ids:
            data_for_current = raw_data_idx.get(current_id)
            if not data_for_current:
                raise ValueError(
                    f"No data found for region_id={current_id}"
                    + f" (mapped from {region_id})."
                )
            raw_data_list_for_region.append(data_for_current)

        if not raw_data_list_for_region:
            raise ValueError(
                f"No data found for region_id={region_id}"
                + f" with current_ids={current_ids}."
            )

        cleaned_data_list_for_region = [
            self.clean(d) for d in raw_data_list_for_region
        ]
        aggr_data = self.get_aggr_data(cleaned_data_list_for_region)
        raw_data_for_region = raw_region_data_idx[region_id]
        mapped_data = (
            dict(
                region_id=region_id,
                region_name=raw_data_for_region["name"],
            )
            | raw_region_data_idx[region_id]
            | aggr_data
        )
        del mapped_data["id"]
        del mapped_data["name"]
        return mapped_data

    def get_data_list(self, regions) -> list[dict]:
        raw_data_list = self.get_base_data_list()
        raw_data_idx = {
            d["region_id"]: d for d in raw_data_list if "region_id" in d
        }

        raw_region_data_idx = {
            r["id"]: r for r in regions.raw_region_data_list
        }

        mapped_data_list = []
        for region_id, current_ids in self.get_where_to_what_id_map(
            regions
        ).items():
            mapped_data = self._get_mapped_data_for_region(
                region_id, current_ids, raw_data_idx, raw_region_data_idx
            )
            mapped_data_list.append(mapped_data)

        return mapped_data_list
