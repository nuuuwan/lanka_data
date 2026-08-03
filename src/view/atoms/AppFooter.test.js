import { render, screen } from "@testing-library/react";
import { ThemeProvider } from "@mui/material/styles";

import { AppTheme } from "../../AppTheme";
import { APP_URL } from "../../nonview/constants/APP";
import AppFooter from "./AppFooter";

test("renders a themed QR code below the author link", () => {
  render(
    <ThemeProvider theme={AppTheme}>
      <AppFooter />
    </ThemeProvider>,
  );

  expect(screen.getByRole("link", { name: "Open Lanka Data" })).toHaveAttribute(
    "href",
    APP_URL,
  );

  const authorLink = screen.getByRole("link", { name: /@nuuuwan/ });
  const links = screen.getAllByRole("link");
  const qrCode = screen.getByTitle("Scan to open Lanka Data");
  const qrCodeLink = screen.getByRole("link", { name: "Open Lanka Data" });

  expect(links.indexOf(authorLink)).toBeLessThan(links.indexOf(qrCodeLink));
  expect(qrCode).toContainHTML(`<path fill="${AppTheme.palette.info.main}"`);
});
