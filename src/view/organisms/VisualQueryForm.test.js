import { fireEvent, render, screen, within } from "@testing-library/react";
import { useState } from "react";

import VisualQueryForm from "./VisualQueryForm.js";

const VISUAL_QUERY = "Person/Time=2024+Province+Religion/Count/BarChart";
const QUERY_OPTIONS = {
  entities: ["House", "Person", "Vote"],
  dimensionsByEntity: {
    Person: ["Time", "Province", "Religion", "District"],
  },
};

function StatefulVisualQueryForm({ onChange = jest.fn() }) {
  const [value, setValue] = useState(VISUAL_QUERY);

  return (
    <VisualQueryForm
      value={value}
      onChange={(nextValue) => {
        setValue(nextValue);
        onChange(nextValue);
      }}
      onSubmit={jest.fn()}
      queryOptions={QUERY_OPTIONS}
    />
  );
}

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

  test("offers metadata-derived data choices", () => {
    render(
      <VisualQueryForm
        value={VISUAL_QUERY}
        onChange={jest.fn()}
        onSubmit={jest.fn()}
        queryOptions={QUERY_OPTIONS}
      />,
    );

    fireEvent.mouseDown(screen.getByLabelText("What data?"));

    const listbox = screen.getByRole("listbox");
    expect(within(listbox).getByRole("option", { name: "House" })).toBeVisible();
    expect(within(listbox).getByRole("option", { name: "Vote" })).toBeVisible();
  });

  test("adds AND conditions with explicit operators", () => {
    const onChange = jest.fn();
    render(<StatefulVisualQueryForm onChange={onChange} />);

    fireEvent.click(screen.getByRole("button", { name: "AND" }));
    const fieldInputs = screen.getAllByLabelText("Field");
    const operatorInputs = screen.getAllByLabelText("Operator");
    const valueInputs = screen.getAllByLabelText("Value");
    const lastIndex = fieldInputs.length - 1;

    fireEvent.change(fieldInputs[lastIndex], { target: { value: "District" } });
    fireEvent.change(operatorInputs[lastIndex], { target: { value: "=" } });
    fireEvent.change(valueInputs[lastIndex], { target: { value: "colombo" } });

    expect(onChange).toHaveBeenLastCalledWith(
      "Person/Time=2024+Province+Religion+District=colombo/Count/BarChart",
    );
  });

  test("groups related visual choices under headings", () => {
    render(
      <VisualQueryForm
        value={VISUAL_QUERY}
        onChange={jest.fn()}
        onSubmit={jest.fn()}
        queryOptions={QUERY_OPTIONS}
      />,
    );

    fireEvent.mouseDown(screen.getByLabelText("Show as"));

    expect(screen.getByText("Charts")).toBeVisible();
    expect(screen.getByText("Maps")).toBeVisible();
    expect(screen.getByText("Other")).toBeVisible();
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
