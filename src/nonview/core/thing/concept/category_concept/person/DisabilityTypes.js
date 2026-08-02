import CategoryConcept from "../CategoryConcept.js";

export default class DisabilityTypes extends CategoryConcept {
  static validValues() {
    return [
      "cognitive_difficulty",
      "difficulty_in_seeing",
      "hearing_difficulty",
      "no_disability",
      "selfcare_difficulty",
      "social_difficulty",
      "walking_difficulty",
    ];
  }

  static mapAlias() {
    return {
      cognitive_difficulty: ["difficulty_in_remembering_or_concentrating"],
      hearing_difficulty: ["difficulty_in_hearing"],
      selfcare_difficulty: [
        "difficulty_in_selfcare_such_as_washing_or_dressing",
      ],
      social_difficulty: ["difficulty_in_communicating_with_others"],
      walking_difficulty: ["difficulty_in_walking_or_climbing_steps"],
    };
  }

  static getColorMap() {
    return {
      cognitive_difficulty: "#D03899",
      difficulty_in_seeing: "#D05D38",
      hearing_difficulty: "#3840D0",
      no_disability: "#cccccc",
      selfcare_difficulty: "#38C5D0",
      social_difficulty: "#D0AF38",
      walking_difficulty: "#6CD038",
    };
  }
}
