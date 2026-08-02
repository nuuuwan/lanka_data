import CategoryConcept from "../CategoryConcept.js";

export default class NonCommunicableDisease extends CategoryConcept {
  static validValues() {
    return [
      "asthma",
      "cancer",
      "diabetes",
      "epilepsy",
      "heart_disease",
      "high_blood_pressure",
      "high_cholesterol",
      "kidney_disease",
      "stroke_or_paralysis",
      "thalassemia",
    ];
  }

  static mapAlias() {
    return {
      stroke_or_paralysis: ["stroke"],
    };
  }

  static getColorMap() {
    return {
      asthma: "#38C5D0",
      cancer: "#38D056",
      diabetes: "#3840D0",
      epilepsy: "#D03847",
      heart_disease: "#D03899",
      high_blood_pressure: "#D05D38",
      high_cholesterol: "#6CD038",
      kidney_disease: "#D0AF38",
      stroke_or_paralysis: "#8238D0",
      thalassemia: "#3873D0",
    };
  }
}
