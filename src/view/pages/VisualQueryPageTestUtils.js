import { render } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";

import RecentVisualQueries from "../../nonview/base/RecentVisualQueries.js";
import DataContext from "../../nonview/core/data_context/DataContext.js";
import DataSourceFactory from "../../nonview/core/data_source/DataSourceFactory.js";
import VisualQuery from "../../nonview/core/VisualQuery.js";
import Person from "../../nonview/core/thing/entity/Person.js";
import VisualQueryPage from "./VisualQueryPage.js";

jest.mock("../../nonview/core/VisualQuery.js", () => ({
  __esModule: true,
  default: { fromString: jest.fn() },
}));
jest.mock("../../nonview/core/data_source/DataSourceFactory.js", () => ({
  __esModule: true,
  default: { getDatumSetForQuery: jest.fn() },
}));

export const VISUAL_QUERY = "Person/Time=2024+Province+Religion/Count/BarChart";
export { DataSourceFactory, Person, RecentVisualQueries, VisualQuery };

const originalScrollIntoView = HTMLElement.prototype.scrollIntoView;
export const scrollIntoView = jest.fn();

beforeEach(() => {
  localStorage.clear();
  jest.spyOn(console, "error").mockImplementation(() => undefined);
  HTMLElement.prototype.scrollIntoView = scrollIntoView;
  scrollIntoView.mockClear();
});
afterEach(() => {
  jest.useRealTimers();
  jest.restoreAllMocks();
  HTMLElement.prototype.scrollIntoView = originalScrollIntoView;
});

export function renderPage(path = "/bad-request") {
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

export function mockVisual(VisualClass, query = {}) {
  VisualQuery.fromString.mockResolvedValue({
    query: {
      aggregate: "Count",
      dimThingList: [],
      entityClass: Person,
      ...query,
    },
    visualClassName: VisualClass.name,
  });
}
