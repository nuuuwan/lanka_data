import CategoryConcept from "../CategoryConcept.js";

export default class ToiletFacilities extends CategoryConcept {
  static validValues() {
    return [
      "common_public",
      "direct_pit",
      "housing_unit_private",
      "housing_unit_shared",
      "no_toilet_open",
      "no_toilet_sharing",
      "not_using_a_toilet",
      "other",
      "pour_flush",
      "premises_exclusive",
      "premises_shared",
      "water_seal_piped",
      "water_seal_septic",
    ];
  }

  static mapAlias() {
    return {
      common_public: ["common_or_public_toilet"],
      housing_unit_private: [
        "within_the_housing_unit_exclusively_for_the_household",
        "within_unit_exclusive",
      ],
      housing_unit_shared: [
        "within_the_housing_unit_sharing_with_another_household",
        "within_unit_shared",
      ],
      no_toilet_open: ["not_using_a_toilet_use_jungle_beach_and_open_ground"],
      no_toilet_sharing: [
        "no_toilet_but_sharing_with_another_housing_unit_or_units",
      ],
      not_using_a_toilet: ["none"],
      pour_flush: ["pour_flush_toilet_not_water_seal"],
      premises_exclusive: [
        "within_premises_exclusive",
        "within_premises_exclusively_for_the_household",
      ],
      premises_shared: [
        "within_premises_shared",
        "within_premises_sharing_with_another_household",
      ],
      water_seal_piped: ["water_seal_and_connected_to_a_piped_sewer_system"],
      water_seal_septic: ["water_seal_and_connected_to_a_septic_tank"],
    };
  }

  static getColorMap() {
    return {
      common_public: "#8238D0",
      direct_pit: "#D03899",
      housing_unit_private: "#9FD038",
      housing_unit_shared: "#3873D0",
      no_toilet_open: "#38C5D0",
      no_toilet_sharing: "#D0AF38",
      not_using_a_toilet: "#cccccc",
      other: "#cccccc",
      pour_flush: "#6CD038",
      premises_exclusive: "#D03847",
      premises_shared: "#38D056",
      water_seal_piped: "#D05D38",
      water_seal_septic: "#3840D0",
    };
  }
}
