import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import TableVisual from "./TableVisual.js";

class District {
  constructor(value) {
    this.value = value;
  }

  getLabel() {
    return this.value;
  }
}

function createDatum(district, value) {
  return {
    query: {
      dimThingList: [new District(district)],
      aggregate: "Count",
    },
    answerThing: { value },
  };
}

test("shows an accessible empty state", () => {
  render(<TableVisual datumSet={{ datumList: [] }} />);

  expect(screen.getByRole("table", { name: "Query results" })).toBeVisible();
  expect(screen.getByText("No query results available.")).toBeVisible();
});

test("renders dimensions and a formatted aggregate for a single row", () => {
  render(
    <TableVisual datumSet={{ datumList: [createDatum("colombo", 1234)] }} />,
  );

  expect(screen.getByRole("columnheader", { name: "District" })).toBeVisible();
  expect(screen.getByRole("columnheader", { name: "Count" })).toBeVisible();
  const row = screen.getAllByRole("row")[1];
  expect(within(row).getByText("Colombo")).toBeVisible();
  expect(within(row).getByText("1.2K")).toBeVisible();
});

test("sorts rows by dimension and aggregate values", () => {
  render(
    <TableVisual
      datumSet={{
        datumList: [
          createDatum("galle", 20),
          createDatum("colombo", 100),
          createDatum("kandy", 3),
        ],
      }}
    />,
  );

  const districtHeader = screen.getByRole("columnheader", {
    name: "District",
  });
  userEvent.click(within(districtHeader).getByRole("button"));
  expect(
    screen
      .getAllByRole("row")
      .slice(1)
      .map((row) => within(row).getAllByRole("cell")[0].textContent),
  ).toEqual(["Colombo", "Galle", "Kandy"]);

  const countHeader = screen.getByRole("columnheader", { name: "Count" });
  userEvent.click(within(countHeader).getByRole("button"));
  expect(
    screen
      .getAllByRole("row")
      .slice(1)
      .map((row) => within(row).getAllByRole("cell")[1].textContent),
  ).toEqual(["3", "20", "100"]);
  userEvent.click(within(countHeader).getByRole("button"));
  expect(
    screen
      .getAllByRole("row")
      .slice(1)
      .map((row) => within(row).getAllByRole("cell")[1].textContent),
  ).toEqual(["100", "20", "3"]);
});
