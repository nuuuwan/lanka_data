import CategoryConcept from "../CategoryConcept.js";

export default class Region extends CategoryConcept {
  static validValues() {
    return [];
  }

  static allowArbitraryValues() {
    return true;
  }
}
