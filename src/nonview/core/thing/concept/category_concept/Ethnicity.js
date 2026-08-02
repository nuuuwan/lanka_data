import CategoryConcept from "./CategoryConcept.js";

export default class Ethnicity extends CategoryConcept {
  static validValues() {
    return [
      "bharatha",
      "burgher",
      "burgher_and_eurasian",
      "european",
      "indian_muslim",
      "indian_tamil",
      "low_country_sinhala",
      "malay",
      "other",
      "sinhala",
      "sri_lanka_chetty",
      "sri_lanka_muslim",
      "sri_lanka_tamil",
      "up_country_sinhala",
      "veddahs",
    ];
  }

  static mapAlias() {
    return {
      indian_tamil: [
        "ind_and_malaiyaga_tamil",
        "ind_tamil",
        "indian_malaiyaga_tamil",
        "indian_tamil_or_malaiyaga_thamilar",
      ],
      low_country_sinhala: ["low_country_sinhalese"],
      other: ["other_eth"],
      sinhala: ["sinhalese"],
      sri_lanka_chetty: ["sl_chetty"],
      sri_lanka_muslim: [
        "sl_moor",
        "sri_lanka_moor_muslim",
        "sri_lanka_moor_or_muslim",
      ],
      sri_lanka_tamil: ["sl_tamil"],
      up_country_sinhala: ["up_country_kandyan_sinhalese"],
      veddahs: ["veddas", "veddha"],
    };
  }

  static getColorMap() {
    return {
      bharatha: "#16a085",
      burgher: "#8e44ad",
      burgher_and_eurasian: "#9b59b6",
      european: "#6c5ce7",
      indian_muslim: "#00897b",
      indian_tamil: "#ff8888",
      low_country_sinhala: "#c0392b",
      malay: "#cccccc",
      other: "#999999",
      sinhala: "#941E32",
      sri_lanka_chetty: "#e67e22",
      sri_lanka_muslim: "#005F56",
      sri_lanka_tamil: "#DF7500",
      up_country_sinhala: "#e74c3c",
      veddahs: "#795548",
    };
  }
}
