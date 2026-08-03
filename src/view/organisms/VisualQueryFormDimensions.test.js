import { fireEvent, render, screen } from "@testing-library/react";

import { StatefulVisualQueryForm } from "./VisualQueryFormTestUtils.js";

test("adds AND conditions with explicit operators", () => {
  const onChange = jest.fn();
  render(<StatefulVisualQueryForm onChange={onChange} />);
  fireEvent.click(screen.getByRole("button", { name: "AND" }));
  let index = screen.getAllByLabelText("Field").length - 1;
  fireEvent.mouseDown(screen.getAllByLabelText("Field")[index]);
  fireEvent.click(screen.getByRole("option", { name: "District" }));
  index = screen.getAllByLabelText("Operator").length - 1;
  fireEvent.mouseDown(screen.getAllByLabelText("Operator")[index]);
  fireEvent.click(screen.getByRole("option", { name: "=" }));
  index = screen.getAllByLabelText("Value").length - 1;
  fireEvent.change(screen.getAllByLabelText("Value")[index], {
    target: { value: "colombo" },
  });
  expect(onChange).toHaveBeenLastCalledWith(
    "Person/Time=2024+Province+Religion+District=colombo/Count/BarChart",
  );
});

test("defaults new conditions to None and hides their value", () => {
  render(<StatefulVisualQueryForm />);
  fireEvent.click(screen.getByRole("button", { name: "AND" }));
  expect(screen.getAllByLabelText("Operator").at(-1)).toHaveTextContent("None");
  expect(screen.getAllByLabelText("Value")).toHaveLength(1);
});

test("offers known category values with their colors", () => {
  const onChange = jest.fn();
  render(<StatefulVisualQueryForm onChange={onChange} />);
  fireEvent.mouseDown(screen.getAllByLabelText("Operator")[2]);
  fireEvent.click(screen.getByRole("option", { name: "=" }));
  fireEvent.mouseDown(screen.getAllByLabelText("Value").at(-1));
  const option = screen.getByRole("option", { name: "buddhist" });
  expect(option).toBeVisible();
  expect(screen.getByTestId("buddhist-color")).toHaveStyle(
    "background-color: #FFBE29",
  );
  fireEvent.click(option);
  expect(onChange).toHaveBeenLastCalledWith(
    "Person/Time=2024+Province+Religion=buddhist/Count/BarChart",
  );
});
