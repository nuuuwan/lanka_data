import CategoryConcept from "../CategoryConcept.js";
import {
  OTHER_WATER_SOURCE_ALIASES,
  PIPE_BORNE_NWSDB,
  PIPE_BORNE_ALIASES,
  WATER_SOURCE_ALIASES,
} from "./SourceOfDrinkingWaterAliases.js";
import {
  WATER_SOURCE_COLORS,
  WATER_SOURCE_VALUES,
} from "./SourceOfDrinkingWaterValues.js";

export default class SourceOfDrinkingWater extends CategoryConcept {
  static PIPE_BORNE_NWSDB = PIPE_BORNE_NWSDB;

  static mapAliasPipeBorne() {
    return PIPE_BORNE_ALIASES;
  }

  static mapAliasOther() {
    return OTHER_WATER_SOURCE_ALIASES;
  }

  static mapAlias() {
    return WATER_SOURCE_ALIASES;
  }

  static validValues() {
    return WATER_SOURCE_VALUES;
  }

  static getColorMap() {
    return WATER_SOURCE_COLORS;
  }
}
