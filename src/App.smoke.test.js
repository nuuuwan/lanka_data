// Smoke tests for key visual query routes.
//
// Run with:
//   npm test -- --testPathPattern=App.smoke
// or as part of the full test suite:
//   CI=true npm test

import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import App from "./App";

const urls = [
  "/lanka_data/Person/Time=2024+Province+Religion/Count/MarimekkoChart",
  "/lanka_data/Person/Time+Province=western+Religion/Count/MarimekkoChart",
];

describe.each(urls)("screen: %s", (url) => {
  test("renders without crashing", async () => {
    render(
      <MemoryRouter initialEntries={[url]}>
        <App />
      </MemoryRouter>,
    );

    await waitFor(() => {
      expect(screen.getByTestId("visual-content")).toBeInTheDocument();
    });
  });
});
