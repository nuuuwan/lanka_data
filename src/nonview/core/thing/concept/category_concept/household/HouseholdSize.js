import CategoryConcept from "../CategoryConcept.js";

export default class HouseholdSize extends CategoryConcept {
  static validValues() {
    return ["0", "1", "2", "3", "4", "5", "6", "7_or_more"];
  }

  static mapAlias() {
    return {
      "7_or_more": ["7_or_over"],
    };
  }

  static isOrdered() {
    return true;
  }

  static getColorMap() {
    return {
      0: "#D05D38",
      1: "#3840D0",
      2: "#6CD038",
      3: "#D03899",
      4: "#38C5D0",
      5: "#D0AF38",
      6: "#8238D0",
      "7_or_more": "#38D056",
    };
  }
}
