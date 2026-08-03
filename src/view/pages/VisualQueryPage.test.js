import { act, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";

import DataContext from "../../nonview/core/data_context/DataContext.js";
import DataSourceFactory from "../../nonview/core/data_source/DataSourceFactory.js";
import RecentVisualQueries from "../../nonview/base/RecentVisualQueries.js";
import VisualQuery from "../../nonview/core/VisualQuery.js";
import Person from "../../nonview/core/thing/entity/Person.js";
import VisualQueryPage from "./VisualQueryPage.js";

const VISUAL_QUERY = "Person/Time=2024+Province+Religion/Count/BarChart";

jest.mock("../../nonview/core/VisualQuery.js", () => ({
  __esModule: true,
  default: { fromString: jest.fn() },
}));
jest.mock("../../nonview/core/data_source/DataSourceFactory.js", () => ({
  __esModule: true,
  default: { getDatumSetForQuery: jest.fn() },
}));

function renderPage(path = "/bad-request") {
  return render(
    <DataContext.Provider
      value={{
        isReady: true,
        queryOptions: { entities: [], dimensionsByEntity: {} },
      }}
    >
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="*" element={<VisualQueryPage />} />
        </Routes>
      </MemoryRouter>
    </DataContext.Provider>,
  );
}

beforeEach(() => {
  localStorage.clear();
  jest.spyOn(console, "error").mockImplementation(() => undefined);
});

afterEach(() => {
  jest.useRealTimers();
  jest.restoreAllMocks();
});

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
  VisualQuery.fromString.mockResolvedValue({
    query: {
      aggregate: "Count",
      dimThingList: [],
      entityClass: Person,
    },
    visualClass: TestVisual,
  });
  DataSourceFactory.getDatumSetForQuery.mockResolvedValue({ datumList: [] });

  renderPage();

  expect(await screen.findByTestId("query-error")).toHaveTextContent(
    "We couldn't find any data for that request.",
  );
  expect(screen.queryByTestId("visual-content")).not.toBeInTheDocument();
  expect(RecentVisualQueries.read()).toEqual([]);
});

test("shows visual loading stages with completion times", async () => {
  let resolveDatumSet;
  function TestVisual() {
    return <div>visual</div>;
  }
  VisualQuery.fromString.mockResolvedValue({
    query: {
      aggregate: "Count",
      dimThingList: [],
      entityClass: Person,
    },
    visualClass: TestVisual,
  });
  DataSourceFactory.getDatumSetForQuery.mockReturnValue(
    new Promise((resolve) => {
      resolveDatumSet = resolve;
    }),
  );

  renderPage();

  const progressList = await screen.findByRole("list", {
    name: "Visual loading progress",
  });
  expect(
    screen.getByRole("dialog", { name: "Loading visual" }),
  ).toBeInTheDocument();
  await waitFor(() => {
    expect(screen.getAllByLabelText("Complete")).toHaveLength(2);
  });
  const [applicationStep, requestStep, dataStep] =
    within(progressList).getAllByRole("listitem");
  expect(applicationStep).toHaveTextContent("0.00 seconds");
  expect(requestStep).toHaveTextContent("seconds");
  expect(dataStep).toHaveTextContent(/\d+\.\d{2} seconds/);
  expect(dataStep).not.toHaveTextContent("In progress");

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
  expect(
    screen.getByRole("button", { name: "Change this view" }),
  ).toBeInTheDocument();
  expect(
    screen.getByRole("heading", { name: "Count of people" }),
  ).toBeInTheDocument();
  expect(screen.getByText(/Population: people/)).toHaveTextContent(
    "Geography: all available geographies",
  );
  await waitFor(() => {
    expect(RecentVisualQueries.read()).toEqual(["bad-request"]);
  });
});

test("shows collapsed query controls after the visual", async () => {
  function TestVisual() {
    return <div>visual result</div>;
  }
  VisualQuery.fromString.mockResolvedValue({
    query: {},
    visualClass: TestVisual,
  });
  let resolveDatumSet;
  DataSourceFactory.getDatumSetForQuery.mockReturnValue(
    new Promise((resolve) => {
      resolveDatumSet = resolve;
    }),
  );

  renderPage(`/${VISUAL_QUERY}`);

  expect(
    screen.queryByRole("button", { name: "Change this view" }),
  ).not.toBeInTheDocument();

  resolveDatumSet({ datumList: [{}], provenance: [] });

  const visual = await screen.findByTestId("visual-content");
  const changeViewButton = screen.getByRole("button", {
    name: "Change this view",
  });
  expect(
    visual.compareDocumentPosition(changeViewButton) &
      Node.DOCUMENT_POSITION_FOLLOWING,
  ).toBeTruthy();
  expect(
    screen.queryByRole("combobox", { name: "What data?" }),
  ).not.toBeInTheDocument();

  act(() => {
    changeViewButton.click();
  });

  expect(
    await screen.findByRole("combobox", { name: "What data?" }),
  ).toBeVisible();
});

test("updates active loading time without showing a negative duration", async () => {
  VisualQuery.fromString.mockReturnValue(new Promise(() => {}));
  jest.useFakeTimers();

  renderPage();

  const requestStep = within(
    await screen.findByRole("list", {
      name: "Visual loading progress",
    }),
  ).getAllByRole("listitem")[1];
  expect(requestStep).toHaveTextContent("0.00 seconds");

  act(() => {
    jest.advanceTimersByTime(1000);
  });

  expect(requestStep).toHaveTextContent(/\d+\.\d{2} seconds/);
  expect(requestStep).not.toHaveTextContent("0.00 seconds");
  expect(requestStep).not.toHaveTextContent("-");
});

test("updates elapsed time while visual data is loading", async () => {
  function TestVisual() {
    return <div>visual</div>;
  }
  VisualQuery.fromString.mockResolvedValue({
    query: {},
    visualClass: TestVisual,
  });
  DataSourceFactory.getDatumSetForQuery.mockReturnValue(new Promise(() => {}));
  jest.useFakeTimers();

  renderPage();

  await waitFor(() => {
    expect(screen.getAllByLabelText("Complete")).toHaveLength(2);
  });
  const dataStep = within(
    screen.getByRole("list", {
      name: "Visual loading progress",
    }),
  ).getAllByRole("listitem")[2];
  expect(dataStep).toHaveTextContent("0.00 seconds");

  act(() => {
    jest.advanceTimersByTime(1000);
  });

  expect(dataStep).toHaveTextContent(/\d+\.\d{2} seconds/);
  expect(dataStep).not.toHaveTextContent("0.00 seconds");
});
