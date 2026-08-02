import CategoryConcept from "../CategoryConcept.js";

export default class HouseholdOccupancy extends CategoryConcept {
  static validValues() {
    return ["closed_or_vacant", "occupied"];
  }

  static mapAlias() {
    return {
      closed_or_vacant: ["permanently_closed_or_vacant"],
    };
  }

  static getColorMap() {
    return {
      closed_or_vacant: "#3840D0",
      occupied: "#D05D38",
    };
  }
}
