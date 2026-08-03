import FilterAltIcon from "@mui/icons-material/FilterAlt";

import KeyValue from "../KeyValue.js";
import { SEMANTIC_PALETTE } from "../../constants/COLORS.js";

export default class Thing {
  static WILDCARD = "*";
  static SPECIAL_VALUE_EXCLUDED_SMALL = "excluded_small";
  static SPECIAL_VALUE_EXCLUDED_SMALL_COLOR = SEMANTIC_PALETTE.neutral;

  static getClassName() {
    return this.className || this.name;
  }

  static getMUIICON() {
    return FilterAltIcon;
  }

  constructor(value) {
    this.value = String(value);
  }

  static fromValue(value) {
    return new this(value);
  }

  getLabel() {
    return this.value;
  }

  getHumanReadableValue() {
    return `${this.constructor.getClassName()}=${this.getLabel()}`;
  }

  getColor() {
    if (this.value === Thing.SPECIAL_VALUE_EXCLUDED_SMALL) {
      return Thing.SPECIAL_VALUE_EXCLUDED_SMALL_COLOR;
    }
    return SEMANTIC_PALETTE.neutral;
  }

  toKeyValue() {
    if (this.value === Thing.WILDCARD) {
      return this.constructor.getClassName();
    }
    return `${this.constructor.getClassName()}${KeyValue.DELIM}${this.value}`;
  }
}
