import CategoryConcept from "../CategoryConcept.js";

export default class HighestEducationLevel extends CategoryConcept {
  static validValues() {
    return [
      "degree_and_above",
      "gce_a_or_l",
      "gce_advanced_level",
      "gce_o_or_l",
      "gce_ordinary_level",
      "no_schooling",
      "passed_grade_1_5",
      "passed_grade_6_8",
      "passed_grade_9_10",
      "primary",
      "secondary",
    ];
  }

  static mapAlias() {
    return {
      gce_a_or_l: ["g_c_e_a_or_l_or_equal", "g_c_e_a_or_l_or_equivalent"],
      gce_o_or_l: ["g_c_e_o_or_l_or_equal", "g_c_e_o_or_l_or_equivalent"],
      no_schooling: ["never_attended_school"],
    };
  }

  static getColorMap() {
    return {
      degree_and_above: "#38C5D0",
      gce_a_or_l: "#9FD038",
      gce_advanced_level: "#D03899",
      gce_o_or_l: "#3873D0",
      gce_ordinary_level: "#6CD038",
      no_schooling: "#D0AF38",
      passed_grade_1_5: "#8238D0",
      passed_grade_6_8: "#38D056",
      passed_grade_9_10: "#D03847",
      primary: "#D05D38",
      secondary: "#3840D0",
    };
  }
}
