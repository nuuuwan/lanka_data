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

  const menuItems = screen.getAllByRole("menuitem");
  const recentQuery = menuItems[0];
  const firstExampleQuery = menuItems[2];

  expect(screen.getByText("Recent queries")).toBeVisible();
  expect(screen.getByText("Example queries")).toBeVisible();
  expect(recentQuery).toHaveTextContent(query);
  expect(recentQuery).toHaveTextContent(new Date(timestamp).toLocaleString());
  expect(firstExampleQuery).toHaveTextContent(EXAMPLE_QUERIES[0].label);
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

  const menuItem =
    _kind === "recent"
      ? screen.getAllByRole("menuitem")[0]
      : screen.getByRole("menuitem", { name: /Religions in Colombo/i });
  fireEvent.click(menuItem);

  expect(screen.getByLabelText("Current path")).toHaveTextContent(`/${query}`);
});
