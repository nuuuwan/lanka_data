import CategoryConcept from "../CategoryConcept.js";

export default class EducationActivity extends CategoryConcept {
  static validValues() {
    return [
      "degree_or_postgrad",
      "not_studying",
      "other_education",
      "preschool_education",
      "school_education",
      "vocational_training",
    ];
  }

  static mapAlias() {
    return {
      degree_or_postgrad: [
        "degree_or_postgraduate_education",
        "undergraduate_or_postgraduate_education",
      ],
      other_education: ["other_educational_activity"],
      preschool_education: ["pre_school"],
      vocational_training: ["vocational_training_or_technical_education"],
    };
  }

  static getColorMap() {
    return {
      degree_or_postgrad: "#6CD038",
      not_studying: "#cccccc",
      other_education: "#cccccc",
      preschool_education: "#D05D38",
      school_education: "#3840D0",
      vocational_training: "#D03899",
    };
  }
}
