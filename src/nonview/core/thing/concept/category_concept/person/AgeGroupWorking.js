import CategoryConcept from "../CategoryConcept.js";

export default class AgeGroupWorking extends CategoryConcept {
  static validValues() {
    return ["age_20_64", "age_65_above", "age_below_20"];
  }

  static getColorMap() {
    return {
      age_20_64: "#3840D0",
      age_65_above: "#6CD038",
      age_below_20: "#D05D38",
    };
  }
}
