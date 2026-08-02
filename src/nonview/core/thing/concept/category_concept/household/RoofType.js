import CategoryConcept from "../CategoryConcept.js";

export default class RoofType extends CategoryConcept {
  static validValues() {
    return [
      "asbestos",
      "cadjan_palmyrah",
      "concrete",
      "metal_sheet",
      "not_relevant",
      "other",
      "tile",
      "zink_aluminium_sheet",
    ];
  }

  static mapAlias() {
    return {
      cadjan_palmyrah: ["cadjan_or_palmyrah_or_straw", "cadjan_palmyrah_straw"],
    };
  }

  static getColorMap() {
    return {
      asbestos: "#3840D0",
      cadjan_palmyrah: "#D0AF38",
      concrete: "#6CD038",
      metal_sheet: "#38C5D0",
      not_relevant: "#cccccc",
      other: "#cccccc",
      tile: "#D05D38",
      zink_aluminium_sheet: "#D03899",
    };
  }
}
