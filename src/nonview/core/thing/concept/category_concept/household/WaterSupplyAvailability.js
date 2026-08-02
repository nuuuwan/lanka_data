import CategoryConcept from "../CategoryConcept.js";

export default class WaterSupplyAvailability extends CategoryConcept {
  static validValues() {
    return ["water_all_year", "water_shortage"];
  }

  static mapAlias() {
    return {
      water_all_year: ["households_with_water_supply_throughout_the_year"],
      water_shortage: [
        "households_with_no_water_suppply_for_at_least_one_month",
      ],
    };
  }

  static getColorMap() {
    return {
      water_all_year: "#D05D38",
      water_shortage: "#3840D0",
    };
  }
}
