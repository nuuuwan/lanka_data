import { screen } from "@testing-library/react";

import {
  DataSourceFactory,
  mockVisual,
  renderPage,
} from "./VisualQueryPageTestUtils.js";

test("shows an answer-first header with secondary query details", async () => {
  function TestVisual() {
    return <div>visual</div>;
  }
  const query = {
    aggregate: "Count",
    entityClass: { getClassName: () => "Person" },
    dimThingList: [
      {
        value: "*",
        constructor: { getClassName: () => "Religion" },
      },
    ],
  };
  mockVisual(TestVisual, query);
  DataSourceFactory.getDatumSetForQuery.mockResolvedValue({
    datumList: [{ query }],
    provenance: [],
  });

  renderPage("/Person/Religion/Count/BarChart");

  const finding = await screen.findByRole("heading", {
    level: 1,
    name: "Count of people by religion",
  });
  const visual = screen.getByText("visual");
  expect(finding.compareDocumentPosition(visual)).toBe(
    Node.DOCUMENT_POSITION_FOLLOWING,
  );
  expect(screen.getByText(/Person\/Religion\/Count\/BarChart/)).toBeVisible();
  expect(screen.getByTestId("datums-count")).toHaveTextContent("1 datum");
});
