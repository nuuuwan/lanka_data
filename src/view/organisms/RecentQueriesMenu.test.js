import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";

import RecentVisualQueries from "../../nonview/base/RecentVisualQueries.js";
import RecentQueriesMenu from "./RecentQueriesMenu.js";

function CurrentPath() {
  return <div data-testid="current-path">{useLocation().pathname}</div>;
}

beforeEach(() => {
  localStorage.clear();
});

test("reopens a stored query by navigating to its route", () => {
  const query = "Person/Time=2024+Province/Count/BarChart";
  RecentVisualQueries.add(query);

  render(
    <MemoryRouter initialEntries={["/current-query"]}>
      <RecentQueriesMenu />
      <Routes>
        <Route path="*" element={<CurrentPath />} />
      </Routes>
    </MemoryRouter>,
  );

  fireEvent.click(screen.getByRole("button", { name: "Recent queries" }));
  fireEvent.click(screen.getByRole("menuitem", { name: query }));

  expect(screen.getByTestId("current-path")).toHaveTextContent(`/${query}`);
});
