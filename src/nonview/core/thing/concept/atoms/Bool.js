import ToggleOnIcon from "@mui/icons-material/ToggleOn";

import Concept from "../Concept.js";

export default class Bool extends Concept {
  static getMUIICON() {
    return ToggleOnIcon;
  }

  static fromValue(value) {
    return new this(Boolean(value));
  }
}
