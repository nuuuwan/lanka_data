import { fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter, useLocation } from "react-router-dom";

import { START_EXPLORING_STEPS } from "../../nonview/constants/StartExploring.js";
import StartExploring from "./StartExploring.js";

function CurrentPath() {
  return <output aria-label="Current path">{useLocation().pathname}</output>;
}

function renderStartExploring() {
  return render(
    <MemoryRouter initialEntries={["/current-query"]}>
      <StartExploring />
      <CurrentPath />
    </MemoryRouter>,
  );
}

test("pairs three curated questions with an interpretation", () => {
  renderStartExploring();

  const sequence = screen.getByRole("list");
  const steps = within(sequence).getAllByRole("listitem");

  expect(steps).toHaveLength(3);
  START_EXPLORING_STEPS.forEach(({ question, interpretation }, index) => {
    expect(
      within(steps[index]).getByRole("link", { name: question }),
    ).toBeVisible();
    expect(steps[index]).toHaveTextContent(interpretation);
  });
});

test("loads the selected query and visual state", () => {
  renderStartExploring();
  const step = START_EXPLORING_STEPS[2];

  fireEvent.click(screen.getByRole("link", { name: step.question }));

  expect(screen.getByLabelText("Current path")).toHaveTextContent(
    `/${step.query}`,
  );
});
