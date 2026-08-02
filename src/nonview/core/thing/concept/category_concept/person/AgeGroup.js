import Concept from "../../Concept.js";
import TimeDurationGroup from "../../time/TimeDurationGroup.js";
import Thing from "../../../Thing.js";

export default class AgeGroup extends Concept {
  static MIN_TIME = TimeDurationGroup.MIN_TIME;
  static MAX_TIME = TimeDurationGroup.MAX_TIME;
  static MORE_WORDS = TimeDurationGroup.MORE_WORDS;
  static LESS_WORDS = TimeDurationGroup.LESS_WORDS;
  static TOTAL_WORDS = ["total", "sri_lanka"];

  constructor(minVal, maxVal) {
    super(`${minVal}To${maxVal}Years`);
    this.minVal = minVal;
    this.maxVal = maxVal;
  }

  static hasTotalTerms(value) {
    const lower = value.toLowerCase();
    return this.TOTAL_WORDS.some((term) => lower.includes(term));
  }

  static hasMoreTerms(value) {
    const lower = value.toLowerCase();
    return this.MORE_WORDS.some((term) => lower.includes(term));
  }

  static hasLessTerms(value) {
    const lower = value.toLowerCase();
    return this.LESS_WORDS.some((term) => lower.includes(term));
  }

  static fromValue(value) {
    if (value === Thing.SPECIAL_VALUE_EXCLUDED_SMALL) {
      return new this(
        Thing.SPECIAL_VALUE_EXCLUDED_SMALL,
        Thing.SPECIAL_VALUE_EXCLUDED_SMALL,
      );
    }

    let normalized = String(value).replace(/[-\s]/g, "_").replace(/To/g, "_");

    if (this.hasTotalTerms(normalized)) {
      return new this(this.MIN_TIME, this.MAX_TIME);
    }

    const numValue = normalized
      .split("")
      .map((c) => (c >= "0" && c <= "9" ? c : " "))
      .join("")
      .replace(/\s+/g, " ")
      .trim();
    const numTokens = numValue.split(" ").map((t) => Number.parseInt(t, 10));

    if (this.hasMoreTerms(normalized)) {
      return new this(numTokens[0], this.MAX_TIME);
    }

    if (this.hasLessTerms(normalized)) {
      return new this(this.MIN_TIME, numTokens[0]);
    }

    if (numTokens.length === 1) {
      return new this(numTokens[0], numTokens[0]);
    }

    if (numTokens.length >= 2) {
      return new this(numTokens[0], numTokens[1]);
    }

    throw new Error(`Cannot parse AgeGroup from value: ${value}`);
  }
}
