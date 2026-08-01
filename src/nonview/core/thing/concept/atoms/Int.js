import Concept from "../Concept.js";

function floorLog10(x) {
  return Math.floor(Math.log10(x));
}

export default class Int extends Concept {
  getHumanReadableValue() {
    let value = parseInt(this.value, 10);
    if (value < 0) {
      return "-" + new Int(-value).getHumanReadableValue();
    }
    if (value < 1000) {
      return String(value);
    }

    for (const [log1000, label] of [
      [1, "K"],
      [2, "M"],
      [3, "B"],
      [4, "T"],
      [5, "P"],
      [6, "E"],
    ]) {
      const threshold = 1000 ** (log1000 + 1);
      if (value < threshold) {
        const mask = 1000 ** log1000;
        const displayValue = value / mask;
        const decimalPlacesInValue = floorLog10(displayValue);
        const decimalPlacesToDisplay = Math.max(0, 1 - decimalPlacesInValue);
        const ndigits = 1 - decimalPlacesInValue;
        const factor = 10 ** ndigits;
        const roundedValue = Math.round(displayValue * factor) / factor;
        return `${roundedValue.toFixed(decimalPlacesToDisplay)}${label}`;
      }
    }

    throw new Error(`Value ${value} is too large to humanize`);
  }
}
