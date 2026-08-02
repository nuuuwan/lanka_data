import CategoryConcept from "../CategoryConcept.js";

export default class Lighting extends CategoryConcept {
  static validValues() {
    return [
      "bio_gas",
      "electricity_grid",
      "generator",
      "kerosene",
      "national_grid",
      "other",
      "rural_hydro",
      "solar_grid",
      "solar_power",
      "solar_standalone",
    ];
  }

  static mapAlias() {
    return {
      kerosene: ["kerosene_lamp"],
      national_grid: ["electricity_national_electricity_network"],
      rural_hydro: ["electricity_rural_hydro_electricity_projects"],
    };
  }

  static getColorMap() {
    return {
      bio_gas: "#38C5D0",
      electricity_grid: "#D0AF38",
      generator: "#D03847",
      kerosene: "#6CD038",
      national_grid: "#D05D38",
      other: "#cccccc",
      rural_hydro: "#3840D0",
      solar_grid: "#8238D0",
      solar_power: "#D03899",
      solar_standalone: "#38D056",
    };
  }
}
