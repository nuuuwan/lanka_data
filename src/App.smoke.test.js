import { render, screen, waitFor } from "@testing-library/react";
import App from "./App";

jest.setTimeout(30000);

const paths = [
  "/lanka_data/Person/Time=2024+Province+Religion/Count/Blocks",
  "/lanka_data/Person/Time=2024+District=Western+Religion/Count/BarChart",
  "/lanka_data/Vote/ElectionType+Time=1994+ED+Party/Count/StackedBarChart",
  "/lanka_data/Vote/ElectionType=presidential+Time=2024+PD<ED=colombo+Party/Count/MarimekkoChart",
];

describe.each(paths)("screen: %s", (path) => {
  const originalLocation = window.location;

  beforeEach(() => {
    // 1. Delete the restricted JSDOM location object
    delete window.location;
    // 2. Assign a new URL object with your target port and path
    window.location = new URL(`http://localhost:3000${path}`);
  });

  afterEach(() => {
    // 3. Restore the original location to prevent leaking state to other tests
    window.location = originalLocation;
  });

  test("renders without crashing", async () => {
    render(<App />);

    await waitFor(
      () => {
        expect(screen.getByText(/datum/)).toBeInTheDocument();
      },
      { timeout: 25_000 },
    );
  });
});
