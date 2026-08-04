export const PIPE_BORNE_NWSDB =
  "pipe_borne_water_national_water_supply_and_drainage_board";

export const PIPE_BORNE_ALIASES = {
  pipe_borne_comm: [
    "pipe_borne_community",
    "pipe_borne_water_community_based_organization",
  ],
  pipe_borne_local: [
    "pipe_borne_local_authority",
    "pipe_borne_water_local_authority",
  ],
  pipe_borne_nwsdb: [PIPE_BORNE_NWSDB],
  pipe_borne_private: ["pipe_borne_water_private_water_supply_project"],
};

export const OTHER_WATER_SOURCE_ALIASES = {
  filter_ro: ["filter_water_r_o_plant"],
  protected_well_in: ["protected_well_within_premises"],
  protected_well_out: ["protected_well_outside_premises"],
  rain_water: ["rainwater"],
  spring_or_fountain: ["spring_fountain"],
  tank_river_stream: ["river_or_tank_or_stream", "tank_or_river_or_streams"],
  tap_outside: ["tap_outside_premises_main_line"],
  tap_outside_unit: ["tap_within_premises_but_outside_unit_main_line"],
  tap_unit_main: ["tap_within_unit_main_line"],
};

export const WATER_SOURCE_ALIASES = {
  ...PIPE_BORNE_ALIASES,
  ...OTHER_WATER_SOURCE_ALIASES,
};
