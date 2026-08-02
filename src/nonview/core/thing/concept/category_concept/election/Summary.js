import Concept from "../../Concept.js";

export default class Summary extends Concept {
  static fromValue(value) {
    return new this(value);
  }
}
