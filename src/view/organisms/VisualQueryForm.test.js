import { fireEvent, screen, within } from "@testing-library/react";

import {
  QUERY_OPTIONS,
  renderForm,
  VISUAL_QUERY,
} from "./VisualQueryFormTestUtils.js";

test("switches between layperson and expert modes", () => {
  renderForm();
  expect(
    screen.getByRole("combobox", { name: "What data?" }),
  ).toHaveTextContent("Person");
  fireEvent.click(screen.getByRole("button", { name: "Expert Mode" }));
  expect(screen.getByLabelText("Visual query")).toHaveValue(VISUAL_QUERY);
});

test("updates a query part without changing the other parts", () => {
  const onChange = jest.fn();
  renderForm({ onChange });
  fireEvent.change(screen.getByLabelText("Calculate"), {
    target: { value: "Total" },
  });
  expect(onChange).toHaveBeenCalledWith(
    "Person/Time=2024+Province+Religion/Total/BarChart",
  );
});

test("offers metadata-derived data choices", () => {
  renderForm({ queryOptions: QUERY_OPTIONS });
  fireEvent.mouseDown(screen.getByLabelText("What data?"));
  const listbox = screen.getByRole("listbox");
  expect(within(listbox).getByRole("option", { name: "House" })).toBeVisible();
  expect(within(listbox).getByRole("option", { name: "Vote" })).toBeVisible();
});

test("groups related visual choices under headings", () => {
  renderForm();
  fireEvent.mouseDown(screen.getByLabelText("Show as"));
  expect(screen.getByText("Charts")).toBeVisible();
  expect(screen.getByText("Maps")).toBeVisible();
  expect(screen.getByText("Other")).toBeVisible();
});
