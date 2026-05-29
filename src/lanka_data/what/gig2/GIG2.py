from lanka_data.what.What import What
from utils_future import WWW, JSONFile


class GIG2(What):

    def __init__(self, title: str, region_group: str, year: str):
        super().__init__(title)
        self.region_group = region_group
        self.year = year

    @classmethod
    def get_title_to_id(cls):
        return JSONFile(cls.get_title_to_id_file_path()).read()

    @classmethod
    def clean(cls, d, region_idx):
        region_id = d["entity_id"]

        for k, v in d.items():
            if "total" in k:
                continue
            if "entity_id" in k:
                continue

        return dict(
            region_id=region_id,
            region_name=region_idx[region_id]["name"],
        ) | cls.get_custom_data(d)

    def get_data_list(self, regions) -> list[dict]:
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

        region_idx = {r["id"]: r for r in regions.raw_region_data_list}
        filtered_data_list = [
            d for d in data_list if d["entity_id"] in region_idx
        ]
        cleaned_data_list = [
            self.clean(d, region_idx) for d in filtered_data_list
        ]
        return cleaned_data_list
