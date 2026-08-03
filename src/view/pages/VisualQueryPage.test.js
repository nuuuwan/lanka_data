import { screen } from "@testing-library/react";

import {
  DataSourceFactory,
  mockVisual,
  RecentVisualQueries,
  renderPage,
  VisualQuery,
} from "./VisualQueryPageTestUtils.js";

test("shows a friendly message when a request cannot be understood", async () => {
  VisualQuery.fromString.mockRejectedValue(new Error("Invalid query"));
  renderPage();
  expect(await screen.findByTestId("query-error")).toHaveTextContent(
    "We couldn't understand that request.",
  );
  expect(screen.queryByText("Invalid query")).not.toBeInTheDocument();
});

test("shows a friendly message when a request returns no data", async () => {
  function TestVisual() {
    return <div>visual</div>;
  }
  mockVisual(TestVisual);
  DataSourceFactory.getDatumSetForQuery.mockResolvedValue({ datumList: [] });
  renderPage();
  expect(await screen.findByTestId("query-error")).toHaveTextContent(
    "We couldn't find any data for that request.",
  );
  expect(screen.queryByTestId("visual-content")).not.toBeInTheDocument();
  expect(RecentVisualQueries.read()).toEqual([]);
});
