import CategoryConcept from "../CategoryConcept.js";

export default class CensusTopic extends CategoryConcept {
  static validValues() {
    return [
      "age",
      "all_literacy",
      "citizenship",
      "clergy_or_priest",
      "computer_literacy",
      "date_of_birth",
      "demographic_info",
      "digital_literacy",
      "education_attainment",
      "education_chars",
      "english_literacy",
      "ethnic_group",
      "literacy",
      "marital_status",
      "n_i_c_no",
      "name",
      "relationship_head",
      "religion",
      "schedule",
      "school_attendance",
      "sex",
      "speak_all_languages",
      "speak_english",
      "speak_sinhala_tamil",
      "vocational_quals",
    ];
  }

  static mapAlias() {
    return {
      all_literacy: ["sinhala_english_and_tamil_literacy"],
      clergy_or_priest: ["status_of_clergy_or_priest"],
      demographic_info: ["demographic_and_personal_information"],
      education_attainment: ["educational_attainment_or_highest_level_of"],
      education_chars: ["educational_characteristics"],
      relationship_head: ["relationship_to_head_of_the_household"],
      school_attendance: ["school_attendance_or_attend_in_educational"],
      speak_all_languages: ["ability_to_speak_sinhala_english_and_tamil"],
      speak_english: ["ability_to_speak_english"],
      speak_sinhala_tamil: ["ability_to_speak_sinhala_and_tamil"],
      vocational_quals: ["vocational_and_apprenticeship_qualification"],
    };
  }

  static getColorMap() {
    return {
      age: "#8238D0",
      all_literacy: "#A238D0",
      citizenship: "#9FD038",
      clergy_or_priest: "#38D0A8",
      computer_literacy: "#38D076",
      date_of_birth: "#D0AF38",
      demographic_info: "#3840D0",
      digital_literacy: "#D04938",
      education_attainment: "#3853D0",
      education_chars: "#D07C38",
      english_literacy: "#D0CE38",
      ethnic_group: "#D03847",
      literacy: "#38A6D0",
      marital_status: "#38D056",
      n_i_c_no: "#D038CB",
      name: "#6CD038",
      relationship_head: "#D03899",
      religion: "#3873D0",
      schedule: "#D05D38",
      school_attendance: "#80D038",
      sex: "#38C5D0",
      speak_all_languages: "#D03879",
      speak_english: "#4DD038",
      speak_sinhala_tamil: "#5038D0",
      vocational_quals: "#D038AC",
    };
  }
}
