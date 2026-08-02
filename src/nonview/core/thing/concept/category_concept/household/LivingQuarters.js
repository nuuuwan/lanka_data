import CategoryConcept from "../CategoryConcept.js";

export default class LivingQuarters extends CategoryConcept {
  static validValues() {
    return ["collective_quarter", "housing_unit", "non_housing_unit"];
  }

  static mapAlias() {
    return {
      collective_quarter: ["collective_living_quarter"],
    };
  }

  static getColorMap() {
    return {
      collective_quarter: "#3840D0",
      housing_unit: "#D05D38",
      non_housing_unit: "#6CD038",
    };
  }
}
