import { render, screen } from "@testing-library/react";

import VisualErrorBoundary from "./VisualErrorBoundary.js";

function BrokenVisual() {
  throw new Error("Broken visual");
}

test("shows a friendly message when a visualization crashes", () => {
  const consoleError = jest
    .spyOn(console, "error")
    .mockImplementation(() => undefined);

  render(
    <VisualErrorBoundary>
      <BrokenVisual />
    </VisualErrorBoundary>,
  );

  expect(screen.getByTestId("query-error")).toHaveTextContent(
    "Sorry, we couldn't show this visualization.",
  );
  expect(screen.queryByText("Broken visual")).not.toBeInTheDocument();

  consoleError.mockRestore();
});
