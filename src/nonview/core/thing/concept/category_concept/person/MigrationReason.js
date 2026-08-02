import CategoryConcept from "../CategoryConcept.js";

export default class MigrationReason extends CategoryConcept {
  static validValues() {
    return [
      "development_projects",
      "disaster_displaced",
      "education",
      "family_accompanied",
      "job_search",
      "marriage",
      "other",
      "permanent_return",
      "resettled",
    ];
  }

  static mapAlias() {
    return {
      disaster_displaced: [
        "a_disaster_a_displaced_happened_in_the_prior_place",
      ],
      family_accompanied: ["accompanied_a_family_member"],
      job_search: ["employment_searching_for_job"],
      permanent_return: ["returning_for_permanent_residence"],
      resettled: ["resettled_after_displacement"],
    };
  }

  static getColorMap() {
    return {
      development_projects: "#D0AF38",
      disaster_displaced: "#38D056",
      education: "#6CD038",
      family_accompanied: "#D03899",
      job_search: "#3840D0",
      marriage: "#D05D38",
      other: "#cccccc",
      permanent_return: "#38C5D0",
      resettled: "#8238D0",
    };
  }
}
