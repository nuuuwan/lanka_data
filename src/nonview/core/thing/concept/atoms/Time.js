import ScheduleIcon from "@mui/icons-material/Schedule";

import Concept from "../Concept.js";

export default class Time extends Concept {
  static getMUIICON() {
    return ScheduleIcon;
  }

  static fromValue(value) {
    return new this(String(value).slice(-4));
  }
}
