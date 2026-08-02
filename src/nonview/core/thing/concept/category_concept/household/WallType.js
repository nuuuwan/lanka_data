import CategoryConcept from "../CategoryConcept.js";

export default class WallType extends CategoryConcept {
  static validValues() {
    return [
      "bricks",
      "cabook",
      "cadjan_palmyrah",
      "cement_block",
      "cement_block_stone",
      "granite_cube_stones",
      "not_relevant",
      "other",
      "plank_or_metal_sheet",
      "planks_metal",
      "pressed_soil_bricks",
      "warichchi_mud",
      "zink_aluminium",
    ];
  }

  static mapAlias() {
    return {
      bricks: ["brick"],
      cadjan_palmyrah: ["cadjan_or_palmyrah"],
      cement_block_stone: ["cement_block_or_stone"],
      planks_metal: ["planks_metal_sheets_asbestos"],
      pressed_soil_bricks: ["soil_bricks"],
      warichchi_mud: ["mud"],
      zink_aluminium: ["zink_aluminium_sheets"],
    };
  }

  static getColorMap() {
    return {
      bricks: "#cc0000",
      cabook: "#ff8800",
      cadjan_palmyrah: "#D0AF38",
      cement_block: "#cccccc",
      cement_block_stone: "#888888",
      granite_cube_stones: "#000000",
      not_relevant: "#cccccc",
      other: "#cccccc",
      plank_or_metal_sheet: "#8238D0",
      planks_metal: "#3873D0",
      pressed_soil_bricks: "#D03899",
      warichchi_mud: "#cc8800",
      zink_aluminium: "#38C5D0",
    };
  }
}
