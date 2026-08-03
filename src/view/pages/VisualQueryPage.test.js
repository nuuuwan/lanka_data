import { render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";

import DataContext from "../../nonview/core/data_context/DataContext.js";
import DataSourceFactory from "../../nonview/core/data_source/DataSourceFactory.js";
import VisualQuery from "../../nonview/core/VisualQuery.js";
import VisualQueryPage from "./VisualQueryPage.js";

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
  jest.spyOn(console, "error").mockImplementation(() => undefined);
});

afterEach(() => {
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
    query: {},
    visualClass: TestVisual,
  });
  DataSourceFactory.getDatumSetForQuery.mockResolvedValue({ datumList: [] });

  renderPage();

  expect(await screen.findByTestId("query-error")).toHaveTextContent(
    "We couldn't find any data for that request.",
  );
  expect(screen.queryByTestId("visual-content")).not.toBeInTheDocument();
});

test("shows visual loading stages with completion times", async () => {
  let resolveDatumSet;
  function TestVisual() {
    return <div>visual</div>;
  }
  VisualQuery.fromString.mockResolvedValue({
    query: {},
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

  resolveDatumSet({ datumList: [{}] });

  expect(await screen.findByTestId("visual-content")).toBeInTheDocument();
});
