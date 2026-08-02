import CategoryConcept from "../CategoryConcept.js";

export default class ResidentRelativeToDistrict extends CategoryConcept {
  static validValues() {
    return ["in_district", "in_other_district"];
  }

  static getColorMap() {
    return {
      in_district: "#D05D38",
      in_other_district: "#3840D0",
    };
  }
}
