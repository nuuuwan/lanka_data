import CategoryConcept from "../CategoryConcept.js";

export default class EmmigrationReason extends CategoryConcept {
  static validValues() {
    return ["education", "employment", "family_in_need", "other"];
  }

  static mapAlias() {
    return {
      family_in_need: ["accompanying_family_member_in_need"],
    };
  }

  static getColorMap() {
    return {
      education: "#3840D0",
      employment: "#D05D38",
      family_in_need: "#6CD038",
      other: "#cccccc",
    };
  }
}
