import Int from "../../../nonview/core/thing/concept/atoms/Int.js";
import Party from "../../../nonview/core/thing/concept/category_concept/election/Party.js";

export default class FormatUtils {
  static isLightColor(color) {
    const hex = color.replace("#", "");
    const red = parseInt(hex.substring(0, 2), 16) / 255;
    const green = parseInt(hex.substring(2, 4), 16) / 255;
    const blue = parseInt(hex.substring(4, 6), 16) / 255;
    const luminance = 0.299 * red + 0.587 * green + 0.114 * blue;
    return luminance > 0.4;
  }

  static humanizeValue(value) {
    const int = new Int(value);
    return int.getHumanReadableValue();
  }

  static humanizeDuration(seconds) {
    return `${seconds.toFixed(2)} seconds`;
  }

  static toThingLabel(thing) {
    if (thing instanceof Party) {
      return thing.getLabel();
    }
    return FormatUtils.toTitleCase(thing.getLabel());
  }

  static toTitleCase(value) {
    return value
      .split("_")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(" ");
  }
}
