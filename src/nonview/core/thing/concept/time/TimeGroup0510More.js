import CategoryConcept from "../category_concept/CategoryConcept.js";

export default class TimeGroup0510More extends CategoryConcept {
  static validValues() {
    return ["00_to_04_years", "05_to_09_years", "10_or_more_years"];
  }

  static getColorMap() {
    return {
      "00_to_04_years": "#D05D38",
      "05_to_09_years": "#3840D0",
      "10_or_more_years": "#6CD038",
    };
  }
}
