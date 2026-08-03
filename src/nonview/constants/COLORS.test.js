import Thing from "../core/thing/Thing.js";
import { getMarkColor, SEMANTIC_PALETTE } from "./COLORS.js";

test("uses neutral grey for unhighlighted marks", () => {
  expect(getMarkColor()).toBe(SEMANTIC_PALETTE.neutral);
  expect(new Thing("unhighlighted").getColor()).toBe(SEMANTIC_PALETTE.neutral);
});

test("preserves category colours that direct attention", () => {
  expect(getMarkColor(SEMANTIC_PALETTE.positive)).toBe(
    SEMANTIC_PALETTE.positive,
  );
});
