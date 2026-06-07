from lanka_data.what.What import What
from utils_future import WWW, JSONFile, Log

log = Log("GIG2")


class GIG2(What):

    def __init__(self, title: str, region_group: str, year: str):
        super().__init__(title)
        self.region_group = region_group
        self.year = year

    @classmethod
    def get_title_to_id(cls):
        return JSONFile(cls.get_title_to_id_file_path()).read()

    def get_source_data_list(self) -> list[dict]:
        title_to_id = self.get_title_to_id()
        what_id = title_to_id.get(self.title)
        if what_id is None:
            raise ValueError(f"Invalid what label: {self.title}")
        url = (
            "https://raw.githubusercontent.com"
            + "/nuuuwan/gig-data/refs/heads/master"
            + f"/gig2/{what_id}.{self.region_group}.{self.year}.tsv"
        )
        data_list = WWW(url).read_tsv()
        return data_list

    def get_data_list(self, regions) -> list[dict]:
        base_data_list = self.get_source_data_list()
        base_data_idx = {d["entity_id"]: d for d in base_data_list}

        region_idx = {r["id"]: r for r in regions.raw_region_data_list}
        region_to_current_ids = regions.region_to_current_ids
        merged_data_list = []
        for region_id, current_ids in region_to_current_ids.items():
            log.debug(f"{region_id} -> {current_ids}")
            data_list = []
            for current_id in current_ids:
                d = base_data_idx.get(current_id)
                if d is not None:
                    data_list.append(d)

            if not data_list:
                continue
            cleaned_data_list = [self.get_custom_data(d) for d in data_list]
            aggr_data = self.get_aggregated_value_data(cleaned_data_list)
            region_data = (
                dict(
                    region_id=region_id,
                    region_name=region_idx[region_id]["name"],
                    current_ids=current_ids,
                )
                | aggr_data
            )
            merged_data_list.append(region_data)

        return merged_data_list
