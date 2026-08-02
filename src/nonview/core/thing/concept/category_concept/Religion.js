import CategoryConcept from "./CategoryConcept.js";

export default class Religion extends CategoryConcept {
  static validValues() {
    return [
      "buddhist",
      "hindu",
      "islam",
      "other",
      "other_christian",
      "roman_catholic",
    ];
  }

  static getColorMap() {
    return {
      buddhist: "#FFBE29",
      hindu: "#DF7500",
      islam: "#005F56",
      other: "#cccccc",
      other_christian: "#2980b9",
      roman_catholic: "#8e44ad",
    };
  }
}
