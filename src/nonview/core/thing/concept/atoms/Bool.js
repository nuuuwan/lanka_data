import Concept from "../Concept.js";

export default class Bool extends Concept {
  static fromValue(value) {
    return new this(Boolean(value));
  }
}
