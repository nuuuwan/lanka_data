import CategoryConcept from "../CategoryConcept.js";

export default class AdministrativeEntity extends CategoryConcept {
  static validValues() {
    return [
      "asst_govt_divisions",
      "gs_divisions",
      "municipal_councils",
      "town_councils",
      "urban_councils",
    ];
  }

  static mapAlias() {
    return {
      asst_govt_divisions: ["assistant_government_agend_divisions"],
      gs_divisions: ["grama_sevaka_divisions"],
    };
  }

  static getColorMap() {
    return {
      asst_govt_divisions: "#D05D38",
      gs_divisions: "#3840D0",
      municipal_councils: "#6CD038",
      town_councils: "#38C5D0",
      urban_councils: "#D03899",
    };
  }
}
