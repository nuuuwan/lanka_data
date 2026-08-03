import { fireEvent, render, screen } from "@testing-library/react";

import VisualQueryForm from "./VisualQueryForm.js";

const VISUAL_QUERY = "Person/Time=2024+Province+Religion/Count/BarChart";

test("switches between layperson and expert modes", () => {
  render(
    <VisualQueryForm
      value={VISUAL_QUERY}
      onChange={jest.fn()}
      onSubmit={jest.fn()}
    />,
  );

  expect(screen.getByLabelText("What data?")).toHaveValue("Person");

  fireEvent.click(screen.getByRole("button", { name: "Expert Mode" }));

  expect(screen.getByLabelText("Visual query")).toHaveValue(VISUAL_QUERY);
});

test("updates a layperson query part without changing the other parts", () => {
  const onChange = jest.fn();
  render(
    <VisualQueryForm
      value={VISUAL_QUERY}
      onChange={onChange}
      onSubmit={jest.fn()}
    />,
  );

  fireEvent.change(screen.getByLabelText("Calculate"), {
    target: { value: "Total" },
  });

  expect(onChange).toHaveBeenCalledWith(
    "Person/Time=2024+Province+Religion/Total/BarChart",
  );
});

test("submits expert queries on Enter while allowing shifted line breaks", () => {
  const onSubmit = jest.fn();
  render(
    <VisualQueryForm
      value={VISUAL_QUERY}
      onChange={jest.fn()}
      onSubmit={onSubmit}
    />,
  );
  fireEvent.click(screen.getByRole("button", { name: "Expert Mode" }));

  const input = screen.getByLabelText("Visual query");
  fireEvent.keyDown(input, { key: "Enter" });
  fireEvent.keyDown(input, { key: "Enter", shiftKey: true });

  expect(onSubmit).toHaveBeenCalledTimes(1);
});

test("submits layperson changes with the update button", () => {
  const onSubmit = jest.fn();
  render(
    <VisualQueryForm
      value={VISUAL_QUERY}
      onChange={jest.fn()}
      onSubmit={onSubmit}
    />,
  );

  fireEvent.click(screen.getByRole("button", { name: "Update" }));

  expect(onSubmit).toHaveBeenCalledTimes(1);
});
