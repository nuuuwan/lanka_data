from lanka_data.datasets.data.FieldNameUtils import FieldNameUtils
from lanka_data.datasets.dataset.custom.ElectionDataset import ElectionDataset
from utils_future import Log

log = Log("ElectionDataset")


class ElectionSummaryDataset(ElectionDataset):
    @classmethod
    def supports(cls, label, year):
        base_label = label.replace("Summary", "")
        return label.endswith("Summary") and base_label in cls.get_labels()

    @classmethod
    def from_summary_label_data_and_year(
        cls, label: str, region_data_list: list[dict], year: str
    ) -> "ElectionSummaryDataset":
        return cls.from_label_and_region_data_list_and_year(
            label.replace("Summary", ""), region_data_list, year
        )

    def clean_data_row(self, row: dict) -> dict:
        d = {"region_id": row["entity_id"]}
        raw_values = {}
        for k, v in row.items():
            if k in ["electors", "polled", "valid", "rejected"]:
                raw_values[FieldNameUtils.normalize(k)] = int(float(v))

        values = {
            "Valid": raw_values.get("Valid", 0),
            "Rejected": raw_values.get("Rejected", 0),
            "DidNotVote": raw_values.get("Electors", 0)
            - raw_values.get("Polled", 0),
        }

        d["values"] = values
        return d
