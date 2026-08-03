import { fireEvent, render, screen } from "@testing-library/react";

import MultiChartLayout from "./MultiChartLayout.js";

const FACETS = [
  { facetKey: "Western", data: "western-data" },
  { facetKey: "Central", data: "central-data" },
];

test("shows the first facet and lets readers select another facet", () => {
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
  expect(screen.queryByText("central-data")).not.toBeInTheDocument();

  fireEvent.mouseDown(screen.getByRole("combobox", { name: "Facet" }));
  fireEvent.click(screen.getByRole("option", { name: "Central" }));

  expect(screen.getByRole("heading", { name: "Central" })).toBeInTheDocument();
  expect(screen.getByText("central-data")).toBeInTheDocument();
  expect(screen.queryByText("western-data")).not.toBeInTheDocument();
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
