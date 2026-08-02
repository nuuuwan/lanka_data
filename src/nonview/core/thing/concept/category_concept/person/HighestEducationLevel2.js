import CategoryConcept from "../CategoryConcept.js";

export default class HighestEducationLevel2 extends CategoryConcept {
  static validValues() {
    return [
      "degree_or_above",
      "never_attended",
      "passed_gce_a_or_l",
      "passed_gce_o_or_l",
      "passed_grade_1_5",
      "passed_grade_6_8",
      "passed_grade_9_10",
      "special_school",
    ];
  }

  static mapAlias() {
    return {
      never_attended: ["never_attended_school"],
      special_school: ["studied_in_a_special_school_or_special_unit"],
    };
  }

  static getColorMap() {
    return {
      degree_or_above: "#38D056",
      never_attended: "#D05D38",
      passed_gce_a_or_l: "#8238D0",
      passed_gce_o_or_l: "#D0AF38",
      passed_grade_1_5: "#6CD038",
      passed_grade_6_8: "#D03899",
      passed_grade_9_10: "#38C5D0",
      special_school: "#3840D0",
    };
  }
}
