import Country from "./region/Country.js";
import Province from "./region/Province.js";
import District from "./region/District.js";
import DSD from "./region/DSD.js";
import GND from "./region/GND.js";

import ED from "./region/ED.js";
import PD from "./region/PD.js";

export default class RegionFactory {
  static Country = Country;
  static Province = Province;
  static District = District;
  static DSD = DSD;
  static GND = GND;

  static ED = ED;
  static PD = PD;

  static list() {
    return [Country, Province, District, DSD, GND, ED, PD];
  }
}
