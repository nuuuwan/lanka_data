from lanka_data.data.FieldNameUtils import FieldNameUtils
from lanka_data.dataset.custom.ElectionDataset import ElectionDataset
from utils_future import Log

log = Log("ElectionDataset")


class ElectionSummaryDataset(ElectionDataset):
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
