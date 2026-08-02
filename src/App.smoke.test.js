import { render, screen, waitFor } from "@testing-library/react";
import App from "./App";

const paths = [
  "/lanka_data/Person/Time=2024+Province+Religion/Count/MarimekkoChart",
  "/lanka_data/Person/Time=2024+District=Western+Religion/Count/MarimekkoChart",
  "/lanka_data/Vote/ElectionType+Time=1994+ED+Party/Count/MarimekkoChart",
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
    console.debug("window.location.href", window.location.href);

    render(<App />);

    await waitFor(() => {
      expect(
        screen.getByRole("heading", { name: "Lanka Data" }),
      ).toBeInTheDocument();
    });
  });
});
