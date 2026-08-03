import { ThemeProvider, createTheme } from "@mui/material/styles";
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useLocation } from "react-router-dom";
import { MemoryRouter } from "react-router-dom";

import { EXAMPLE_QUERIES } from "../../nonview/constants/ExampleQueries.js";
import ExampleQueryGallery from "./ExampleQueryGallery.js";

const TEST_THEME = createTheme({
  components: {
    MuiButtonBase: {
      defaultProps: { disableRipple: true },
    },
  },
});

function CurrentPath() {
  return <output aria-label="Current path">{useLocation().pathname}</output>;
}

function renderGallery() {
  return render(
    <ThemeProvider theme={TEST_THEME}>
      <MemoryRouter initialEntries={["/current-query"]}>
        <ExampleQueryGallery />
        <CurrentPath />
      </MemoryRouter>
    </ThemeProvider>,
  );
}

test("shows all curated example labels when expanded", () => {
  renderGallery();

  fireEvent.click(screen.getByRole("button", { name: /Example Queries/i }));

  EXAMPLE_QUERIES.forEach(({ label }) => {
    expect(
      screen.getByRole("button", { name: new RegExp(label) }),
    ).toBeVisible();
  });
});

test("navigates to an example through the query route", () => {
  renderGallery();
  fireEvent.click(screen.getByRole("button", { name: /Example Queries/i }));

  fireEvent.click(
    screen.getByRole("button", { name: /Religions in Colombo/i }),
  );

  expect(screen.getByLabelText("Current path")).toHaveTextContent(
    `/${EXAMPLE_QUERIES[1].query}`,
  );
});

test("supports keyboard-only expansion and selection", () => {
  renderGallery();

  userEvent.tab();
  expect(
    screen.getByRole("button", { name: /Example Queries/i }),
  ).toHaveFocus();
  userEvent.keyboard("{enter}");

  userEvent.tab();
  expect(
    screen.getByRole("button", { name: /2024 census by province/i }),
  ).toHaveFocus();
  userEvent.keyboard("{enter}");

  expect(screen.getByLabelText("Current path")).toHaveTextContent(
    `/${EXAMPLE_QUERIES[0].query}`,
  );
});
