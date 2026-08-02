import CategoryConcept from "../CategoryConcept.js";

export default class MaritalStatus extends CategoryConcept {
  static validValues() {
    return [
      "divorced",
      "legally_separated",
      "married",
      "married_customary",
      "married_registered",
      "never_married",
      "not_stated",
      "separated_not_legal",
      "widowed",
    ];
  }

  static mapAlias() {
    return {
      separated_not_legal: ["separated_not_legally"],
    };
  }

  static getColorMap() {
    return {
      divorced: "#D0AF38",
      legally_separated: "#8238D0",
      married: "#D03899",
      married_customary: "#6CD038",
      married_registered: "#3840D0",
      never_married: "#D05D38",
      not_stated: "#cccccc",
      separated_not_legal: "#38D056",
      widowed: "#38C5D0",
    };
  }
}
