import CategoryConcept from "../CategoryConcept.js";

export default class Sector extends CategoryConcept {
  static validValues() {
    return ["estate", "estate_rural", "estate_urban", "rural", "urban"];
  }

  static getColorMap() {
    return {
      estate: "#6CD038",
      estate_rural: "#D03899",
      estate_urban: "#38C5D0",
      rural: "#3840D0",
      urban: "#D05D38",
    };
  }
}
