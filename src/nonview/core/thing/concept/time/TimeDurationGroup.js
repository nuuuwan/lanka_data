import Concept from "../Concept.js";

export default class TimeDurationGroup extends Concept {
  static MORE_WORDS = [
    "more",
    "over",
    "greater",
    "than",
    "at least",
    "minimum",
  ];
  static LESS_WORDS = ["less", "under", "fewer", "than", "at most", "maximum"];
  static MAX_TIME = 125;
  static MIN_TIME = 0;

  constructor(minValue, maxValue) {
    super(`${minValue}To${maxValue}Years`);
    if (Number.isNaN(minValue) || Number.isNaN(maxValue)) {
      throw new Error("minValue and maxValue must be numbers");
    }
    if (minValue > maxValue) {
      throw new Error("minValue cannot be greater than maxValue");
    }
    this.minValue = minValue;
    this.maxValue = maxValue;
  }

  static numPart(value) {
    const cleaned = value.replace(/[^0-9_]/g, "");
    return cleaned
      .split("_")
      .filter((t) => t)
      .map((t) => Number.parseInt(t, 10));
  }

  static fromValue(value) {
    const normalized = String(value).replace(/To/g, "_");
    const numTokens = this.numPart(normalized);

    for (const word of this.MORE_WORDS) {
      if (normalized.includes(word)) {
        return new this(numTokens[0], this.MAX_TIME);
      }
    }

    for (const word of this.LESS_WORDS) {
      if (normalized.includes(word)) {
        return new this(this.MIN_TIME, numTokens[0]);
      }
    }

    if (numTokens.length >= 2) {
      return new this(numTokens[0], numTokens[1]);
    }

    throw new Error(`Cannot parse TimeDurationGroup from value: ${value}`);
  }
}
