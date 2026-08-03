import { act, screen } from "@testing-library/react";

import {
  DataSourceFactory,
  mockVisual,
  renderPage,
  VISUAL_QUERY,
} from "./VisualQueryPageTestUtils.js";

test("shows collapsed query controls after the visual", async () => {
  function TestVisual() {
    return <div>visual result</div>;
  }
  mockVisual(TestVisual);
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
  const button = screen.getByRole("button", { name: "Change this view" });
  expect(
    visual.compareDocumentPosition(button) & Node.DOCUMENT_POSITION_FOLLOWING,
  ).toBeTruthy();
  expect(
    screen.queryByRole("combobox", { name: "What data?" }),
  ).not.toBeInTheDocument();
  act(() => button.click());
  expect(
    await screen.findByRole("combobox", { name: "What data?" }),
  ).toBeVisible();
});
