import { render, screen } from "@testing-library/react";

import DataProvenancePanel from "./DataProvenancePanel.js";

test("shows only a single data source with a safe external link", () => {
  render(
    <DataProvenancePanel
      provenance={[
        {
          source: "The Elections Commission of Sri Lanka",
          title: "Sri Lanka election results",
          url: "https://elections.gov.lk",
        },
      ]}
    />,
  );

  expect(
    screen.getByRole("heading", { name: "About this data" }),
  ).toBeInTheDocument();
  expect(
   screen.queryByText("Sri Lanka election results"),
  ).not.toBeInTheDocument();
  expect(
    screen.getByRole("link", {
      name: "The Elections Commission of Sri Lanka",
    }),
  ).toHaveAttribute("target", "_blank");
  expect(
    screen.getByRole("link", {
      name: "The Elections Commission of Sri Lanka",
    }),
  ).toHaveAttribute("rel", "noopener noreferrer");
});

test("shows mixed data sources only and omits unavailable URLs", () => {
  render(
    <DataProvenancePanel
      provenance={[
        {
          source: "Department of Census and Statistics of Sri Lanka",
          title: "Census of Population and Housing 2024",
          url: "https://www.statistics.gov.lk",
        },
        {
          source: "Archived census source",
          title: "Census of Population and Housing 2012",
        },
      ]}
    />,
  );

  expect(
    screen.queryByText("Census of Population and Housing 2024"),
  ).not.toBeInTheDocument();
  expect(
    screen.queryByText("Census of Population and Housing 2012"),
  ).not.toBeInTheDocument();
  expect(screen.getByText("Archived census source")).toBeInTheDocument();
  expect(
    screen.queryByRole("link", { name: "Archived census source" }),
  ).not.toBeInTheDocument();
});
