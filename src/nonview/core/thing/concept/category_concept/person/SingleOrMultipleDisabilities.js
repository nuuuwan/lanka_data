import CategoryConcept from "../CategoryConcept.js";

export default class SingleOrMultipleDisabilities extends CategoryConcept {
  static validValues() {
    return ["multi_disability", "no_disability", "single_disability"];
  }

  static mapAlias() {
    return {
      multi_disability: [
        "multiple_disabilities",
        "with_more_than_one_disability",
      ],
      single_disability: ["with_single_disability"],
    };
  }

  static getColorMap() {
    return {
      multi_disability: "#3840D0",
      no_disability: "#cccccc",
      single_disability: "#D05D38",
    };
  }
}
