import District from "./District.js";
import Region from "./region/Region.js";

export default class GND extends Region {
  static SUB_REGION_OF = District;
  static SUB_REGION_ID_KEY = "district_id";
}
