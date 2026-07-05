import os

from rapidfuzz import fuzz

from lanka_data.datasets.region import RegionRawDataMixin
from lanka_data.api.utils_future import WWW, JSONFile, Log

log = Log("oneoff_lg_correction_map")


_NAME_CORRECTIONS = {
    "Manmunai West (Vavunativu) PS": "Manmunai West PS",
    "Porativepattu PS": "Porthivu Pattu PS",
}


def _correct_name(name):
    return _NAME_CORRECTIONS.get(name, name)


def _find_best_fuzzy_match(name, old_name_to_d):
    scores = [
        (old_name, fuzz.ratio(name, old_name)) for old_name in old_name_to_d
    ]
    return max(scores, key=lambda x: x[1])


def _find_exact_matches(new_name_to_d, old_name_to_d):
    matches = {}
    for new_name, d in new_name_to_d.items():
        corrected = _correct_name(new_name)
        if corrected in old_name_to_d:
            old_id = old_name_to_d[corrected]["id"]
            matches[old_id] = d["region_id"]
            log.debug(f"✅ {d['region_id']}: Exact ")
    return matches


def _find_fuzzy_matches(new_name_to_d, old_name_to_d):
    matches = {}
    for new_name, d in new_name_to_d.items():
        corrected = _correct_name(new_name)
        if corrected in old_name_to_d:
            continue
        new_id = d["region_id"]
        best_old_name, best_score = _find_best_fuzzy_match(
            corrected, old_name_to_d
        )
        if best_score >= 80:
            old_id = old_name_to_d[best_old_name]["id"]
            matches[old_id] = new_id
            log.debug(
                f"☑️ {new_id}: "
                + f'"{corrected}" ~= "{best_old_name}" ({best_score:.1f})'
            )
        else:
            raise ValueError(f"❌ {new_id}: No good match.")
    return matches


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
    old_id_to_new_id = _find_exact_matches(new_name_to_d, old_name_to_d)
    old_id_to_new_id.update(_find_fuzzy_matches(new_name_to_d, old_name_to_d))
    return old_id_to_new_id


if __name__ == "__main__":
    corrections_path = os.path.join(
        "src",
        "lanka_data",
        "datasets",
        "dataset",
        "custom",
        "lg.corrections.json",
    )
    corrections_file = JSONFile(corrections_path)
    corrections_file.write(get_old_id_to_new_id())
    log.info(f"Wrote {corrections_file}")
