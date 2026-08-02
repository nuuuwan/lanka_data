import CategoryConcept from "../CategoryConcept.js";

export default class OccupationStatus extends CategoryConcept {
  static validValues() {
    return ["occupied", "vacant"];
  }

  static getColorMap() {
    return {
      occupied: "#D05D38",
      vacant: "#3840D0",
    };
  }
}
