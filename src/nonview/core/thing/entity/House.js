import HomeIcon from "@mui/icons-material/Home";

import Entity from "./Entity.js";

export default class House extends Entity {
  static getMUIICON() {
    return HomeIcon;
  }
}
