import CategoryConcept from "../CategoryConcept.js";

export default class TypeOfUnit extends CategoryConcept {
  static validValues() {
    return [
      "improvised",
      "not_permanent",
      "permanent",
      "semi_permanent",
      "unclassified",
    ];
  }

  static getColorMap() {
    return {
      improvised: "#D03899",
      not_permanent: "#3840D0",
      permanent: "#D05D38",
      semi_permanent: "#6CD038",
      unclassified: "#cccccc",
    };
  }
}
