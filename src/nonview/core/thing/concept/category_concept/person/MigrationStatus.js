import CategoryConcept from "../CategoryConcept.js";

export default class MigrationStatus extends CategoryConcept {
  static validValues() {
    return ["foreign", "local", "migrant"];
  }

  static getColorMap() {
    return {
      foreign: "#3840D0",
      local: "#D05D38",
      migrant: "#6CD038",
    };
  }
}
