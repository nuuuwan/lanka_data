import Concept from "../Concept.js";

export default class Float extends Concept {
  static fromValue(value) {
    return new this(Number.parseFloat(value));
  }
}
