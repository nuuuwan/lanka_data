import CategoryConcept from "../CategoryConcept.js";

export default class IsEconomicallyActive extends CategoryConcept {
  static validValues() {
    return ["economically_active", "employed", "inactive", "unemployed"];
  }

  static mapAlias() {
    return {
      inactive: ["economically_inactive", "economically_not_active"],
    };
  }

  static getColorMap() {
    return {
      economically_active: "#6CD038",
      employed: "#D05D38",
      inactive: "#D03899",
      unemployed: "#3840D0",
    };
  }
}
