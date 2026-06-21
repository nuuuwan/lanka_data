import os

from rapidfuzz import fuzz

from lanka_data import RegionRawDataMixin
from utils_future import WWW, JSONFile, Log

log = Log("oneoff_lg_correction_map")


def get_old_id_to_new_id():
    url_old = (
        "https://raw.githubusercontent.com"
        + "/nuuuwan/gig-data/refs/heads/master/ents/lg.tsv"
    )
    d_list_old = WWW(url_old).read_tsv()
    old_name_to_d = {d["name"]: d for d in d_list_old}
    d_list_new = RegionRawDataMixin._get_raw_region_data_list_for_region_type(
        "lg", "Current"
    )
    new_name_to_d = {
        d["region_name"].split("/")[0].strip(): d for d in d_list_new
    }

    old_id_to_new_id = {}
    for new_name, d in new_name_to_d.items():
        new_name = {
            "Manmunai West (Vavunativu) PS": "Manmunai West PS",
            "Porativepattu PS": "Porthivu Pattu PS",
        }.get(new_name, new_name)
        new_id = d["region_id"]
        if new_name in old_name_to_d:
            old_d = old_name_to_d[new_name]
            old_id = old_d["id"]
            old_id_to_new_id[old_id] = new_id
            log.debug(f"✅ {new_id}: Exact ")

    for new_name, d in new_name_to_d.items():
        new_name = {
            "Manmunai West (Vavunativu) PS": "Manmunai West PS",
            "Porativepattu PS": "Porthivu Pattu PS",
        }.get(new_name, new_name)
        new_id = d["region_id"]
        if new_name in old_name_to_d:
            continue
        old_name_and_score = []
        for old_name, old_d in old_name_to_d.items():
            fuzz_ratio = fuzz.ratio(new_name, old_name)
            old_name_and_score.append((old_name, fuzz_ratio))

        old_name_and_score.sort(key=lambda x: x[1], reverse=True)
        best_old_name, best_score = old_name_and_score[0]
        if best_score >= 80:
            old_id = old_name_to_d[best_old_name]["id"]
            old_id_to_new_id[old_id] = new_id
            log.debug(
                f"☑️ {new_id}: "
                + f'"{new_name}" ~= "{best_old_name}" ({best_score:.1f})'
            )
        else:
            raise ValueError(f"❌ {new_id}: No good match.")

    return old_id_to_new_id


if __name__ == "__main__":
    corrections_path = os.path.join(
        "src", "lanka_data", "dataset", "custom", "lg.corrections.json"
    )
    corrections_file = JSONFile(corrections_path)
    corrections_file.write(get_old_id_to_new_id())
    log.info(f"Wrote {corrections_file}")
