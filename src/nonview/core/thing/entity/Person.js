import PersonIcon from "@mui/icons-material/Person";

import Entity from "./Entity.js";

export default class Person extends Entity {
  static getMUIICON() {
    return PersonIcon;
  }
}
