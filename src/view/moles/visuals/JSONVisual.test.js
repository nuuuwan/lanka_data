import { render, screen } from "@testing-library/react";

import { JSON_DOWNLOAD_FILE_NAME } from "../../../nonview/constants/APP.js";
import JSONVisual from "./JSONVisual.js";

test("renders JSON with a direct download link", () => {
  const datumSet = { datumList: [{ count: 42 }] };
  window.history.pushState(
    {},
    "",
    "/lanka_data/Vote/Time=2024+Party/Count/JSON",
  );

  render(<JSONVisual datumSet={datumSet} />);

  expect(screen.getByText('"count": 42', { exact: false })).toBeInTheDocument();

  const downloadLink = screen.getByRole("link", { name: "Download JSON" });
  expect(downloadLink).toHaveAttribute("download", JSON_DOWNLOAD_FILE_NAME);
  expect(downloadLink).toHaveAttribute(
    "href",
    "http://localhost/lanka_data/Vote/Time=2024+Party/Count/JSON/raw.json",
  );
});
