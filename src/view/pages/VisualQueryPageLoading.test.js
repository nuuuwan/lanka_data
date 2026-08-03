import { screen, waitFor, within } from "@testing-library/react";

import {
  DataSourceFactory,
  mockVisual,
  RecentVisualQueries,
  renderPage,
  scrollIntoView,
} from "./VisualQueryPageTestUtils.js";

test("shows visual loading stages with completion times", async () => {
  let resolveDatumSet;
  function TestVisual() {
    return <div>visual</div>;
  }
  mockVisual(TestVisual);
  DataSourceFactory.getDatumSetForQuery.mockReturnValue(
    new Promise((resolve) => {
      resolveDatumSet = resolve;
    }),
  );
  renderPage();
  expect(
    screen.getByRole("heading", { name: "Lanka Data" }),
  ).toBeInTheDocument();
  const progress = await screen.findByRole("list", {
    name: "Visual loading progress",
  });
  await waitFor(() =>
    expect(screen.getAllByLabelText("Complete")).toHaveLength(2),
  );
  const [applicationStep, requestStep, dataStep] =
    within(progress).getAllByRole("listitem");
  expect(applicationStep).toHaveTextContent("0.00 seconds");
  expect(requestStep).toHaveTextContent("seconds");
  expect(dataStep).toHaveTextContent(/\d+\.\d{2} seconds/);
  resolveDatumSet({
    datumList: [{}],
    provenance: [
      {
        source: "Department of Census and Statistics of Sri Lanka",
        title: "Census of Population and Housing 2024",
        url: "https://www.statistics.gov.lk",
      },
    ],
  });
  expect(await screen.findByTestId("visual-content")).toBeInTheDocument();
  expect(scrollIntoView).toHaveBeenCalledWith({
    behavior: "smooth",
    block: "start",
  });
  expect(
    screen.getByRole("heading", { name: "Count of people" }),
  ).toBeInTheDocument();
  expect(screen.getByText(/Population: people/)).toHaveTextContent(
    "Geography: all available geographies",
  );
  await waitFor(() =>
    expect(RecentVisualQueries.read()).toEqual([
      expect.objectContaining({ query: "bad-request" }),
    ]),
  );
});
