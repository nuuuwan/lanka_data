import CategoryConcept from "../CategoryConcept.js";

export default class LiquidWasteDisposal extends CategoryConcept {
  static validValues() {
    return [
      "closed_pit",
      "natural_water",
      "open_pit",
      "other",
      "piped_sewer",
      "to_a_drain_on_road",
      "within_the_premises",
    ];
  }

  static mapAlias() {
    return {
      closed_pit: ["to_a_properly_closed_pit"],
      natural_water: ["to_a_stream_or_spring_or_river_or_sea"],
      piped_sewer: ["connected_to_a_piped_sewer"],
    };
  }

  static getColorMap() {
    return {
      closed_pit: "#D05D38",
      natural_water: "#38C5D0",
      open_pit: "#3840D0",
      other: "#cccccc",
      piped_sewer: "#D03899",
      to_a_drain_on_road: "#D0AF38",
      within_the_premises: "#6CD038",
    };
  }
}
