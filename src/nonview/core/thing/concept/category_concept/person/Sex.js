import CategoryConcept from "../CategoryConcept.js";

export default class Sex extends CategoryConcept {
  static validValues() {
    return ["both_sexes", "female", "male"];
  }

  static getColorMap() {
    return {
      both_sexes: "#6CD038",
      female: "#3840D0",
      male: "#D05D38",
    };
  }
}
