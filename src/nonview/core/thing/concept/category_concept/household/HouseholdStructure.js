import CategoryConcept from "../CategoryConcept.js";

export default class HouseholdStructure extends CategoryConcept {
  static validValues() {
    return [
      "attached_11_to_19",
      "attached_1st_floor",
      "attached_20_plus",
      "attached_2nd_floor",
      "attached_3_4_floors",
      "attached_5_to_10",
      "attached_house",
      "condominium",
      "flat",
      "hut_or_shanty",
      "other",
      "row_house",
      "single_house_multi",
      "single_storeyed",
      "twin_house",
      "two_storeyed",
    ];
  }

  static mapAlias() {
    return {
      attached_11_to_19: ["attached_house_from_11_to_19_floors"],
      attached_1st_floor: ["attached_house_1st_floor"],
      attached_20_plus: ["attached_house_from_20_floors_or_more"],
      attached_2nd_floor: ["attached_house_2nd_floor"],
      attached_3_4_floors: ["attached_house_from_3_to_4_floors"],
      attached_5_to_10: ["attached_house_from_5_to_10_floors"],
      attached_house: ["attached_house_or_annex"],
      row_house: ["row_house_or_line_room"],
      single_house_multi: [
        "single_house_more_than_2_floors",
        "single_house_more_than_two_storeyed",
      ],
      single_storeyed: [
        "single_house_single_floor",
        "single_house_single_storeyed",
      ],
      two_storeyed: ["single_house_double_floor", "single_house_two_storeyed"],
    };
  }

  static getColorMap() {
    return {
      attached_11_to_19: "#D07C38",
      attached_1st_floor: "#3873D0",
      attached_20_plus: "#5038D0",
      attached_2nd_floor: "#9FD038",
      attached_3_4_floors: "#D038CB",
      attached_5_to_10: "#38D0A8",
      attached_house: "#D03899",
      condominium: "#D0AF38",
      flat: "#38C5D0",
      hut_or_shanty: "#D03847",
      other: "#cccccc",
      row_house: "#38D056",
      single_house_multi: "#6CD038",
      single_storeyed: "#D05D38",
      twin_house: "#8238D0",
      two_storeyed: "#3840D0",
    };
  }
}
