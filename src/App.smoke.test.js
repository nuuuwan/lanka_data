import { render, screen } from "@testing-library/react";
import App from "./App";

jest.setTimeout(120_000);

jest.mock(
  "react-virtualized-auto-sizer",
  () =>
    function AutoSizer({ children }) {
      return children({ width: 800, height: 400 });
    },
);

const screens = [
  {
    path: "/lanka_data/Person/Time=2024+Province+Religion/Count/Blocks",
    readyTestId: "datums-count",
  },
  {
    path: "/lanka_data/Person/Time=2024+District=colombo+Religion/Count/BarChart",
    readyTestId: "datums-count",
  },
  {
    path: "/lanka_data/Vote/ElectionType+Time=1994+ED+Party/Count/StackedBarChart",
    readyTestId: "datums-count",
  },
  {
    path: "/lanka_data/Vote/ElectionType=presidential+Time=2024+PD<ED=colombo+Party/Count/MarimekkoChart",
    readyTestId: "datums-count",
  },
  {
    path: "/lanka_data/Vote/ElectionType=presidential+Time=2024+PD%3CED=colombo+Party/Count/Map",
    readyTestId: "maps",
  },
  {
    path: "/lanka_data/Vote/ElectionType=presidential+Time=2024+PD+Party/Count/Map",
    readyTestId: "maps",
  },
  {
    path: "/lanka_data/Vote/ElectionType=presidential+Time+Country+Party/Count/Map",
    readyTestId: "map-facets",
  },
  {
    path: "/lanka_data/Vote/ElectionType=presidential+Time=2005+ED+Party/Count/Cartogram",
    readyTestId: "cartogram",
  },
  {
    path: "/lanka_data/Vote/ElectionType=presidential+Time=2015+PD%3CED=gampaha+Party/Count/Cartogram",
    readyTestId: "cartogram",
  },
  {
    path: "/lanka_data/Vote/ElectionType=presidential+Time+PD%3CED=colombo+Party/Count/Cartogram",
    readyTestId: "cartogram-facets",
  },
];

describe.each(screens)("screen: $path", ({ path, readyTestId }) => {
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

    expect(
      await screen.findByTestId(readyTestId, {}, { timeout: 40_000 }),
    ).toBeInTheDocument();
  });
});
