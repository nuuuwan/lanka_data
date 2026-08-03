import TableVisual from "../../organisms/TableVisual.js";
import VisualFactory from "./VisualFactory.js";

test("registers the Table visual", () => {
  expect(VisualFactory.list()).toContain("Table");
  expect(VisualFactory.get("Table")).toBe(TableVisual);
});
