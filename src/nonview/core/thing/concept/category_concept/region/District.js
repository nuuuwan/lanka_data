import Province from "./Province.js";
import Region from "./region/Region.js";

export default class District extends Region {
  static SUB_REGION_OF = Province;
  static SUB_REGION_ID_KEY = "province_id";
}
