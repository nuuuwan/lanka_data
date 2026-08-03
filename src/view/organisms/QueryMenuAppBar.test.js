import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, useLocation } from "react-router-dom";

import RecentVisualQueries from "../../nonview/base/RecentVisualQueries.js";
import { EXAMPLE_QUERIES } from "../../nonview/constants/ExampleQueries.js";
import QueryMenuAppBar from "./QueryMenuAppBar.js";

function CurrentPath() {
  return <output aria-label="Current path">{useLocation().pathname}</output>;
}

function renderAppBar() {
  return render(
    <MemoryRouter initialEntries={["/current-query"]}>
      <QueryMenuAppBar />
      <CurrentPath />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  localStorage.clear();
});

test("lists recent queries with timestamps before example queries", () => {
  const query = "Person/Time=2024+Province/Count/BarChart";
  const timestamp = new Date("2026-08-03T07:28:42Z").getTime();
  RecentVisualQueries.add(query, undefined, timestamp);

  renderAppBar();
  fireEvent.click(screen.getByRole("button", { name: "Queries" }));

  const recentHeading = screen.getByText("Recent queries");
  const recentQuery = screen.getByText(query).closest('[role="menuitem"]');
  const exampleHeading = screen.getByText("Example queries");

  expect(recentHeading.compareDocumentPosition(recentQuery)).toBe(
    Node.DOCUMENT_POSITION_FOLLOWING,
  );
  expect(recentQuery.compareDocumentPosition(exampleHeading)).toBe(
    Node.DOCUMENT_POSITION_FOLLOWING,
  );
  expect(recentQuery).toHaveTextContent(new Date(timestamp).toLocaleString());
});

test.each([
  ["recent", "Person/Time=2024+Province/Count/BarChart"],
  ["example", EXAMPLE_QUERIES[1].query],
])("navigates directly to a %s query route", (_kind, query) => {
  if (_kind === "recent") {
    RecentVisualQueries.add(query);
  }
  renderAppBar();
  fireEvent.click(screen.getByRole("button", { name: "Queries" }));

  const accessibleName =
    _kind === "recent"
      ? screen.getByText(query).closest('[role="menuitem"]')
      : screen.getByRole("menuitem", { name: /Religions in Colombo/i });
  fireEvent.click(accessibleName);

  expect(screen.getByLabelText("Current path")).toHaveTextContent(`/${query}`);
});
