import CategoryConcept from "../CategoryConcept.js";

export default class SolidWasteDisposal extends CategoryConcept {
  static validValues() {
    return [
      "dispose_outdoors",
      "local_collected",
      "occupants_burn",
      "occupants_bury",
      "occupants_compost",
      "other",
    ];
  }

  static mapAlias() {
    return {
      dispose_outdoors: [
        "occupants_dispose_solid_waste_into_road_or_river_or_canal_or_sea_or_creek_or_forest_etc",
      ],
      local_collected: ["collected_by_local_authorities"],
      occupants_compost: ["occupants_composting_solid_waste"],
    };
  }

  static getColorMap() {
    return {
      dispose_outdoors: "#38C5D0",
      local_collected: "#D05D38",
      occupants_burn: "#3840D0",
      occupants_bury: "#6CD038",
      occupants_compost: "#D03899",
      other: "#cccccc",
    };
  }
}
