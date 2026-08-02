import CategoryConcept from "../CategoryConcept.js";

export default class SourceOfDrinkingWater extends CategoryConcept {
  static PIPE_BORNE_NWSDB =
    "pipe_borne_water_national_water_supply_and_drainage_board";

  static mapAliasPipeBorne() {
    return {
      pipe_borne_comm: [
        "pipe_borne_community",
        "pipe_borne_water_community_based_organization",
      ],
      pipe_borne_local: [
        "pipe_borne_local_authority",
        "pipe_borne_water_local_authority",
      ],
      pipe_borne_nwsdb: [this.PIPE_BORNE_NWSDB],
      pipe_borne_private: ["pipe_borne_water_private_water_supply_project"],
    };
  }

  static mapAliasOther() {
    return {
      filter_ro: ["filter_water_r_o_plant"],
      protected_well_in: ["protected_well_within_premises"],
      protected_well_out: ["protected_well_outside_premises"],
      rain_water: ["rainwater"],
      spring_or_fountain: ["spring_fountain"],
      tank_river_stream: [
        "river_or_tank_or_stream",
        "tank_or_river_or_streams",
      ],
      tap_outside: ["tap_outside_premises_main_line"],
      tap_outside_unit: ["tap_within_premises_but_outside_unit_main_line"],
      tap_unit_main: ["tap_within_unit_main_line"],
    };
  }

  static mapAlias() {
    return {
      ...this.mapAliasPipeBorne(),
      ...this.mapAliasOther(),
    };
  }

  static validValues() {
    return [
      "bottled_water",
      "bowser",
      "filter_ro",
      "ground_water",
      "other",
      "outside_premises",
      "pipe_borne_comm",
      "pipe_borne_local",
      "pipe_borne_nwsdb",
      "pipe_borne_private",
      "pipe_borne_water",
      "protected_well",
      "protected_well_in",
      "protected_well_out",
      "rain_water",
      "rural_water_projects",
      "semi_protected_well",
      "spring_or_fountain",
      "tank_river_stream",
      "tap_outside",
      "tap_outside_unit",
      "tap_unit_main",
      "tube_well",
      "unprotected_well",
      "within_housing_unit",
      "within_premises",
    ];
  }

  static getColorMap() {
    return {
      bottled_water: "#D038CB",
      bowser: "#D03847",
      filter_ro: "#D038AC",
      ground_water: "#D07C38",
      other: "#cccccc",
      outside_premises: "#D03879",
      pipe_borne_comm: "#D04938",
      pipe_borne_local: "#38D076",
      pipe_borne_nwsdb: "#A238D0",
      pipe_borne_private: "#3853D0",
      pipe_borne_water: "#38D0A8",
      protected_well: "#38A6D0",
      protected_well_in: "#D05D38",
      protected_well_out: "#3840D0",
      rain_water: "#9FD038",
      rural_water_projects: "#8238D0",
      semi_protected_well: "#D0CE38",
      spring_or_fountain: "#80D038",
      tank_river_stream: "#3873D0",
      tap_outside: "#D0AF38",
      tap_outside_unit: "#38C5D0",
      tap_unit_main: "#D03899",
      tube_well: "#38D056",
      unprotected_well: "#6CD038",
      within_housing_unit: "#5038D0",
      within_premises: "#4DD038",
    };
  }
}
