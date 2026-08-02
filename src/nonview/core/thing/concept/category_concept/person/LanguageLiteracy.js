import CategoryConcept from "../CategoryConcept.js";

export default class LanguageLiteracy extends CategoryConcept {
  static validValues() {
    return ["any_language", "english", "sinhala", "tamil"];
  }

  static mapAlias() {
    return {
      any_language: ["at_least_one_language"],
    };
  }

  static getColorMap() {
    return {
      any_language: "#D05D38",
      english: "#D03899",
      sinhala: "#3840D0",
      tamil: "#6CD038",
    };
  }
}
