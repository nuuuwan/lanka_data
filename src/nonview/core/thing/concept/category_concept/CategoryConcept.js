import Concept from "../Concept.js";
export default class CategoryConcept extends Concept {
  static get_color_map() {
    return {};
  }

  getColor() {
    return this.constructor.get_color_map()[this.value];
  }
}
