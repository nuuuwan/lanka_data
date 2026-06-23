class Diversity:
    @staticmethod
    def _compute_diversity(pct_values, is_pew=False):
        if not pct_values:
            return "(No Data)"
        if is_pew:
            # See
            # https://www.pewresearch.org/religion/2026/02/12/religious-diversity-around-the-world/
            pct_values = {
                "Christians": pct_values["RomanCatholic"]
                + pct_values["OtherChristian"],
                "Hindus": pct_values["Hindu"],
                "Muslims": pct_values["Islam"],
                "Buddhists": pct_values["Buddhist"],
                "Jews": 0,
                "ReligiouslyUnaffiliated": 0,
                "OtherReligions": pct_values["Other"],
            }

        values_only = pct_values.values()
        # normalised Herfindahl-Simpson concentration measure
        return (
            10
            * (1 - sum(s**2 for s in values_only))
            / (1 - 1 / len(values_only))
        )

    RDI_BANDS = [
        (7.0, 10.0, "#1d6614", "Very High (≥7.0)"),
        (5.5, 7.0, "#6a9f3a", "High (5.5–7.0)"),
        (3.0, 5.5, "#d4b030", "Moderate (3.0–5.5)"),
        (1.0, 3.0, "#e07030", "Low (1.0–3.0)"),
        (0.0, 1.0, "#c03025", "Very Low (<1.0)"),
    ]

    # flake8: noqa: C901
    @staticmethod
    def _get_diversity_label_and_color(diversity):
        for low, high, color, label in Diversity.RDI_BANDS:
            if low <= diversity <= high:
                return label, color, low, high
        raise ValueError(f"Diversity value {diversity} out of expected range")

    @staticmethod
    def get_region_to_diversity_change(result_data, is_pew=False):

        region_to_diversity1 = Diversity.get_region_to_diversity(
            result_data, is_pew, "pct_values1"
        )
        region_to_diversity2 = Diversity.get_region_to_diversity(
            result_data, is_pew, "pct_values2"
        )

        common_regions = set(region_to_diversity1.keys()) & set(
            region_to_diversity2.keys()
        )
        uncommon_regions = set(region_to_diversity1.keys()) ^ set(
            region_to_diversity2.keys()
        )

        region_to_diversity_change = {}
        for region_id in common_regions:
            diversity_change = (
                (
                    region_to_diversity2[region_id]
                    - region_to_diversity1[region_id]
                )
                if region_to_diversity2[region_id] != "(No Data)"
                and region_to_diversity1[region_id] != "(No Data)"
                else "(No Data)"
            )
            if diversity_change != "(No Data)":
                diversity_change = round(diversity_change, 4)
            region_to_diversity_change[region_id] = diversity_change

        for region_id in uncommon_regions:
            region_to_diversity_change[region_id] = None

        return region_to_diversity_change

    @staticmethod
    def get_region_to_diversity(
        dataset, is_pew=False, pct_values_key="pct_values"
    ):
        data_list = dataset.get_data_table()
        region_to_diversity = {}
        for data in data_list:
            diversity = Diversity._compute_diversity(
                data[pct_values_key], is_pew
            )
            region_to_diversity[data["region_id"]] = diversity
        return region_to_diversity
