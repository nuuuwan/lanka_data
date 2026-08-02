import CategoryConcept from "../CategoryConcept.js";

export default class MigrationLifetimeDirection extends CategoryConcept {
  static validValues() {
    return ["in_migrants", "out_migrants"];
  }

  static getColorMap() {
    return {
      in_migrants: "#D05D38",
      out_migrants: "#3840D0",
    };
  }
}
