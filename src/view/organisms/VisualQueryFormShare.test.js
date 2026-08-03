import { fireEvent, render, screen, waitFor } from "@testing-library/react";

import {
  setClipboard,
  StatefulVisualQueryForm,
  VISUAL_QUERY,
} from "./VisualQueryFormTestUtils.js";

beforeEach(() => {
  window.history.pushState(
    {},
    "",
    `/lanka_data/${VISUAL_QUERY}?view=compact#results`,
  );
});
afterEach(() => {
  setClipboard(undefined);
  delete document.execCommand;
});

test("copies the full visualization URL and confirms success", async () => {
  const writeText = jest.fn().mockResolvedValue(undefined);
  setClipboard({ writeText });
  render(<StatefulVisualQueryForm />);
  fireEvent.click(screen.getByRole("button", { name: "Copy Share Link" }));
  await waitFor(() =>
    expect(writeText).toHaveBeenCalledWith(window.location.href),
  );
  expect(await screen.findByText("Share link copied")).toBeInTheDocument();
});

test("falls back when the Clipboard API is unavailable", async () => {
  setClipboard(undefined);
  document.execCommand = jest.fn().mockReturnValue(true);
  render(<StatefulVisualQueryForm />);
  fireEvent.click(screen.getByRole("button", { name: "Copy Share Link" }));
  await waitFor(() =>
    expect(document.execCommand).toHaveBeenCalledWith("copy"),
  );
  expect(await screen.findByText("Share link copied")).toBeInTheDocument();
});

test("reports failure when the link cannot be copied", async () => {
  setClipboard(undefined);
  document.execCommand = jest.fn().mockReturnValue(false);
  render(<StatefulVisualQueryForm />);
  fireEvent.click(screen.getByRole("button", { name: "Copy Share Link" }));
  expect(
    await screen.findByText("Could not copy share link"),
  ).toBeInTheDocument();
});
