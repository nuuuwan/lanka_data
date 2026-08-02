import CategoryConcept from "../CategoryConcept.js";

export default class HouseholdType extends CategoryConcept {
  static validValues() {
    return ["composite", "extended", "nuclear", "one_person"];
  }

  static getColorMap() {
    return {
      composite: "#D03899",
      extended: "#6CD038",
      nuclear: "#3840D0",
      one_person: "#D05D38",
    };
  }
}
