import Country from "./Country.js";
import Province from "./Province.js";
import District from "./District.js";
import DSD from "./DSD.js";
import GND from "./GND.js";

import ED from "./ED.js";
import PD from "./PD.js";

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
