import CategoryConcept from "../CategoryConcept.js";

export default class CookingFuel extends CategoryConcept {
  static validValues() {
    return [
      "bio_gas",
      "electricity",
      "fire_wood",
      "gas",
      "kerosene",
      "not_relevant",
      "other",
      "sawdust_paddy_husk",
    ];
  }

  static mapAlias() {
    return {
      fire_wood: ["firewood"],
      sawdust_paddy_husk: ["saw_dust_or_paddy_husk"],
    };
  }

  static getColorMap() {
    return {
      bio_gas: "#D0AF38",
      electricity: "#D03899",
      fire_wood: "#D05D38",
      gas: "#6CD038",
      kerosene: "#3840D0",
      not_relevant: "#cccccc",
      other: "#cccccc",
      sawdust_paddy_husk: "#38C5D0",
    };
  }
}
