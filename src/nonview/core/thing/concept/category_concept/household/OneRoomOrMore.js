import CategoryConcept from "../CategoryConcept.js";

export default class OneRoomOrMore extends CategoryConcept {
  static validValues() {
    return ["more_than_one_room", "with_only_one_room"];
  }

  static mapAlias() {
    return {
      more_than_one_room: ["with_only_more_than_one_room"],
    };
  }

  static getColorMap() {
    return {
      more_than_one_room: "#3840D0",
      with_only_one_room: "#D05D38",
    };
  }
}
