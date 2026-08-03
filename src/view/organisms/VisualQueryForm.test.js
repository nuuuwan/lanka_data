import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { useState } from "react";

import VisualQueryForm from "./VisualQueryForm.js";

const VISUAL_QUERY = "Person/Time=2024+Province+Religion/Count/BarChart";
const QUERY_OPTIONS = {
  entities: ["House", "Person", "Vote"],
  dimensionsByEntity: {
    Person: ["Time", "Province", "Religion", "District"],
  },
};

function setClipboard(clipboard) {
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: clipboard,
  });
}

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

  expect(
    screen.getByRole("combobox", { name: "What data?" }),
  ).toHaveTextContent("Person");

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
  let lastIndex = screen.getAllByLabelText("Field").length - 1;
  fireEvent.mouseDown(screen.getAllByLabelText("Field")[lastIndex]);
  fireEvent.click(screen.getByRole("option", { name: "District" }));

  lastIndex = screen.getAllByLabelText("Operator").length - 1;
  fireEvent.mouseDown(screen.getAllByLabelText("Operator")[lastIndex]);
  fireEvent.click(screen.getByRole("option", { name: "=" }));

  lastIndex = screen.getAllByLabelText("Value").length - 1;
  fireEvent.change(screen.getAllByLabelText("Value")[lastIndex], {
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

  const buddhistOption = screen.getByRole("option", { name: "buddhist" });
  expect(buddhistOption).toBeVisible();
  expect(screen.getByTestId("buddhist-color")).toHaveStyle(
    "background-color: #FFBE29",
  );

  fireEvent.click(buddhistOption);
  expect(onChange).toHaveBeenLastCalledWith(
    "Person/Time=2024+Province+Religion=buddhist/Count/BarChart",
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

describe("copy share link", () => {
  beforeEach(() => {
    window.history.pushState(
      {},
      "",
      `/lanka_data/${VISUAL_QUERY}?view=compact#results`,
    );
  });

  afterEach(() => {
    setClipboard(undefined);
    delete document.execCommand;
  });

  test("copies the full visualization URL and confirms success", async () => {
    const writeText = jest.fn().mockResolvedValue(undefined);
    setClipboard({ writeText });
    render(<StatefulVisualQueryForm />);

    fireEvent.click(screen.getByRole("button", { name: "Copy Share Link" }));

    await waitFor(() => {
      expect(writeText).toHaveBeenCalledWith(window.location.href);
    });
    expect(await screen.findByText("Share link copied")).toBeInTheDocument();
  });

  test("falls back when the Clipboard API is unavailable", async () => {
    setClipboard(undefined);
    document.execCommand = jest.fn().mockReturnValue(true);
    render(<StatefulVisualQueryForm />);

    fireEvent.click(screen.getByRole("button", { name: "Copy Share Link" }));

    await waitFor(() => {
      expect(document.execCommand).toHaveBeenCalledWith("copy");
    });
    expect(screen.getByText("Share link copied")).toBeInTheDocument();
  });

  test("reports failure when the link cannot be copied", async () => {
    setClipboard(undefined);
    document.execCommand = jest.fn().mockReturnValue(false);
    render(<StatefulVisualQueryForm />);

    fireEvent.click(screen.getByRole("button", { name: "Copy Share Link" }));

    expect(
      await screen.findByText("Could not copy share link"),
    ).toBeInTheDocument();
  });
});
