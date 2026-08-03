import { render, screen } from "@testing-library/react";
import { ThemeProvider } from "@mui/material/styles";

import { AppTheme } from "../../AppTheme";
import { APP_URL } from "../../nonview/constants/APP";
import AppFooter from "./AppFooter";

test("renders a QR code linking to the application", () => {
  render(
    <ThemeProvider theme={AppTheme}>
      <AppFooter />
    </ThemeProvider>,
  );

  expect(screen.getByRole("link", { name: "Open Lanka Data" })).toHaveAttribute(
    "href",
    APP_URL,
  );
  expect(screen.getByTitle("Scan to open Lanka Data")).toBeInTheDocument();
});
