import Int from "../../../nonview/core/thing/concept/atoms/Int.js";

export default class FormatUtils {
  static humanizeValue(value) {
    const int = new Int(value);
    return int.getHumanReadableValue();
  }

  static toTitleCase(value) {
    return value
      .split("_")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(" ");
  }
}
