from lanka_data.what.What import What
from utils_future import WWW, JSONFile, Log

log = Log("GIG2")


class GIG2(What):

    def __init__(self, title: str, region_group: str, year: str):
        super().__init__(title)
        self.region_group = region_group
        self.year = year

    @classmethod
    def get_region_year(cls):
        return "2019"

    @classmethod
    def get_title_to_id(cls):
        raw = JSONFile(cls.get_title_to_id_file_path()).read()
        return {
            title: (entry["id"] if isinstance(entry, dict) else entry)
            for title, entry in raw.items()
        }

    @classmethod
    def get_title_to_description(cls):
        raw = JSONFile(cls.get_title_to_id_file_path()).read()
        return {
            title: entry.get("description", "")
            for title, entry in raw.items()
            if isinstance(entry, dict)
        }

    @classmethod
    def get_where_to_what_id_map(cls, regions) -> dict:
        idx = {}
        for region in regions.raw_region_data_list:
            region_id = region["region_id"]
            if regions.region_year == "Current":
                current_ids = region.get("current_ids", [region_id])
            else:
                current_ids = [region_id.split("-pre")[0]]

            if region_id == "LK-5221":
                continue

            idx[region_id] = current_ids
        return idx

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

        def remap(d):
            d["region_id"] = d["entity_id"]
            del d["entity_id"]
            return d

        remapped_data_list = [remap(d) for d in data_list]
        return remapped_data_list
