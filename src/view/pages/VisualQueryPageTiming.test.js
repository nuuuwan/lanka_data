import { act, screen, waitFor, within } from "@testing-library/react";

import {
  DataSourceFactory,
  mockVisual,
  renderPage,
  VisualQuery,
} from "./VisualQueryPageTestUtils.js";

function getProgressStep(index) {
  return within(
    screen.getByRole("list", { name: "Visual loading progress" }),
  ).getAllByRole("listitem")[index];
}

test("updates active loading time without negative duration", async () => {
  VisualQuery.fromString.mockReturnValue(new Promise(() => {}));
  jest.useFakeTimers();
  renderPage();
  await screen.findByRole("list", { name: "Visual loading progress" });
  const requestStep = getProgressStep(1);
  expect(requestStep).toHaveTextContent("0.00 seconds");
  act(() => jest.advanceTimersByTime(1000));
  expect(requestStep).toHaveTextContent(/\d+\.\d{2} seconds/);
  expect(requestStep).not.toHaveTextContent("0.00 seconds");
  expect(requestStep).not.toHaveTextContent("-");
});

test("updates elapsed time while visual data is loading", async () => {
  function TestVisual() {
    return <div>visual</div>;
  }
  mockVisual(TestVisual);
  DataSourceFactory.getDatumSetForQuery.mockReturnValue(new Promise(() => {}));
  jest.useFakeTimers();
  renderPage();
  await waitFor(() =>
    expect(screen.getAllByLabelText("Complete")).toHaveLength(2),
  );
  const dataStep = getProgressStep(2);
  expect(dataStep).toHaveTextContent("0.00 seconds");
  act(() => jest.advanceTimersByTime(1000));
  expect(dataStep).toHaveTextContent(/\d+\.\d{2} seconds/);
  expect(dataStep).not.toHaveTextContent("0.00 seconds");
});
