import { fireEvent, render, screen } from "@testing-library/react";

import MultiChartLayout from "./MultiChartLayout.js";

const FACETS = [
  { facetKey: "Western", data: "western-data" },
  { facetKey: "Central", data: "central-data" },
];

test("shows every facet by default and lets readers deselect and reselect them", () => {
  render(
    <MultiChartLayout
      facets={FACETS}
      xAxisDimName="District"
      yAxisLabel="Population"
      renderChart={({ data }) => <div>{data}</div>}
    />,
  );

  expect(screen.getByRole("heading", { name: "Western" })).toBeInTheDocument();
  expect(screen.getByText("western-data")).toBeInTheDocument();
  expect(screen.getByRole("heading", { name: "Central" })).toBeInTheDocument();
  expect(screen.getByText("central-data")).toBeInTheDocument();

  fireEvent.mouseDown(screen.getByRole("combobox", { name: "Facets" }));
  fireEvent.click(screen.getByRole("option", { name: "Central" }));

  expect(
    screen.queryByRole("heading", { name: "Central" }),
  ).not.toBeInTheDocument();
  expect(screen.queryByText("central-data")).not.toBeInTheDocument();
  expect(screen.getByText("western-data")).toBeInTheDocument();

  fireEvent.click(screen.getByRole("option", { name: "Central" }));

  expect(screen.getByRole("heading", { name: "Central" })).toBeInTheDocument();
  expect(screen.getByText("central-data")).toBeInTheDocument();
});

test("does not show a facet selector for a single result", () => {
  render(
    <MultiChartLayout
      facets={[FACETS[0]]}
      xAxisDimName="District"
      yAxisLabel="Population"
      renderChart={({ data }) => <div>{data}</div>}
    />,
  );

  expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
  expect(screen.getByRole("heading", { name: "Western" })).toBeInTheDocument();
});
