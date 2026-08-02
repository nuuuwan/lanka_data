import Concept from "../Concept.js";

export default class Time extends Concept {
  static fromValue(value) {
    return new this(String(value).slice(-4));
  }
}
