import { fireEvent, screen } from "@testing-library/react";

import { renderForm } from "./VisualQueryFormTestUtils.js";

test("submits expert queries on Enter while allowing shifted line breaks", () => {
  const onSubmit = jest.fn();
  renderForm({ onSubmit });
  fireEvent.click(screen.getByRole("button", { name: "Expert Mode" }));
  const input = screen.getByLabelText("Visual query");
  fireEvent.keyDown(input, { key: "Enter" });
  fireEvent.keyDown(input, { key: "Enter", shiftKey: true });
  expect(onSubmit).toHaveBeenCalledTimes(1);
});

test("submits layperson changes with the update button", () => {
  const onSubmit = jest.fn();
  renderForm({ onSubmit });
  fireEvent.click(screen.getByRole("button", { name: "Update" }));
  expect(onSubmit).toHaveBeenCalledTimes(1);
});
