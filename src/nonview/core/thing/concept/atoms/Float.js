import NumbersIcon from "@mui/icons-material/Numbers";

import Concept from "../Concept.js";

export default class Float extends Concept {
  static getMUIICON() {
    return NumbersIcon;
  }

  static fromValue(value) {
    return new this(Number.parseFloat(value));
  }
}
