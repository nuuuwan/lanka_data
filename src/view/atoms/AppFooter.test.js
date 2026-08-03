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
  const qrCode = screen.getByTitle("Scan to open Lanka Data").closest("svg");
  const qrCodeLink = screen.getByRole("link", { name: "Open Lanka Data" });

  expect(
    authorLink.compareDocumentPosition(qrCodeLink) &
      Node.DOCUMENT_POSITION_FOLLOWING,
  ).toBeTruthy();
  expect(qrCode.querySelector("path:last-of-type")).toHaveAttribute(
    "fill",
    AppTheme.palette.info.main,
  );
});
