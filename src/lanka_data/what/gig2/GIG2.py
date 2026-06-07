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

    @classmethod
    def extract_source_data_values(cls, d):
        values = {}
        for k, v in d.items():
            if k in ("region_id", "region_name"):
                continue
            if "total" in k:
                continue
            values[k] = int(float(v))
        return dict(values=values)

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
