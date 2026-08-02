import Thing from "../../../Thing.js";
import Party from "./Party.js";

test("uses the excluded-small color", () => {
  const party = Party.fromValue(Thing.SPECIAL_VALUE_EXCLUDED_SMALL);

  expect(party.getColor()).toBe("#ccc");
});
