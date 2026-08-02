import { render, screen, waitFor } from "@testing-library/react";
import App from "./App";

const paths = [
  "/lanka_data/Person/Time=2024+Province+Religion/Count/MarimekkoChart",
  "/lanka_data/Person/Time=2024+District=Western+Religion/Count/MarimekkoChart",
  "/lanka_data/Vote/ElectionType+Time=1994+ED+Party/Count/MarimekkoChart",
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
    window.fetch = jest.fn((url) => {
      if (typeof url === "string" && url.includes("/ents/provinces.json")) {
        return Promise.resolve({
          json: () => Promise.resolve([{ id: "LK-1", name: "Western" }]),
        });
      }
      if (typeof url === "string" && url.includes("/ents/districts.json")) {
        return Promise.resolve({
          json: () =>
            Promise.resolve([
              { id: "LK-1", name: "Western", province_id: "LK-1" },
              { id: "LK-11", name: "Colombo", province_id: "LK-1" },
            ]),
        });
      }
      if (typeof url === "string" && url.includes("/ents/eds.json")) {
        return Promise.resolve({
          json: () =>
            Promise.resolve([
              { id: "LK-1100", name: "Colombo", district_id: "LK-11" },
            ]),
        });
      }
      if (typeof url === "string" && url.includes("/ents/pds.json")) {
        return Promise.resolve({
          json: () =>
            Promise.resolve([
              {
                id: "LK-1100001",
                name: "Colombo PD",
                ed_id: "LK-1100",
              },
            ]),
        });
      }
      return Promise.resolve({
        json: () => Promise.resolve([]),
      });
    });

    render(<App />);

    await waitFor(() => {
      expect(
        screen.getByRole("heading", { name: "Lanka Data" }),
      ).toBeInTheDocument();
    });
  });
});
