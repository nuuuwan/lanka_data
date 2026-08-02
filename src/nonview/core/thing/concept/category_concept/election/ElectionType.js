import CategoryConcept from "../CategoryConcept.js";

export default class ElectionType extends CategoryConcept {
  static validValues() {
    return ["local_government", "parliamentary", "presidential"];
  }

  static getColorMap() {
    return {
      local_government: "#6CD038",
      parliamentary: "#D05D38",
      presidential: "#3840D0",
    };
  }
}
