import CategoryConcept from "../CategoryConcept.js";

export default class OwnershipStatus extends CategoryConcept {
  static validValues() {
    return [
      "encroached",
      "occupied_free",
      "other",
      "owned_by_household",
      "rent_free",
      "rent_government",
      "rent_private",
    ];
  }

  static mapAlias() {
    return {
      occupied_free: ["occupied_free_of_rent"],
      owned_by_household: ["owned_by_a_household_member"],
      rent_free: ["rent_or_lease_free_of_rent"],
      rent_government: ["rent_or_lease_government_owned"],
      rent_private: ["rent_or_lease_privately_owned"],
    };
  }

  static getColorMap() {
    return {
      encroached: "#38C5D0",
      occupied_free: "#D03899",
      other: "#cccccc",
      owned_by_household: "#D05D38",
      rent_free: "#D0AF38",
      rent_government: "#3840D0",
      rent_private: "#6CD038",
    };
  }
}
