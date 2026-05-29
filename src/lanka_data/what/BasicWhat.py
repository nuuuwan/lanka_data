from lanka_data.what.What import What


class BasicWhat(What):

    def __init__(self):
        super().__init__(title="Basic Information")

    @classmethod
    def clean(cls, data):
        new_data = {}
        for k, v in data.items():
            if k == "id":
                new_data["region_id"] = v
            elif k == "name":
                new_data["region_name"] = v
            else:
                new_data[k] = v
        return new_data

    def get_data_list(self, regions) -> list[dict]:
        return [self.clean(region) for region in regions.raw_region_data_list]

    def get_source_info(self) -> dict:
        return dict(
            source="Department of Census and Statistics, Sri Lanka",
            source_url="https://www.statistics.gov.lk/",
        )
