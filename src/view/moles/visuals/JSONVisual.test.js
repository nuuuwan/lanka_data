import { render, screen } from "@testing-library/react";

import {
  JSON_DATA_URL_PREFIX,
  JSON_DOWNLOAD_FILE_NAME,
} from "../../../nonview/constants/APP.js";
import JSONVisual from "./JSONVisual.js";

test("renders JSON with a direct download link", () => {
  const datumSet = { datumList: [{ count: 42 }] };

  render(<JSONVisual datumSet={datumSet} />);

  expect(screen.getByText('"count": 42', { exact: false })).toBeInTheDocument();

  const downloadLink = screen.getByRole("link", { name: "Download JSON" });
  expect(downloadLink).toHaveAttribute("download", JSON_DOWNLOAD_FILE_NAME);
  expect(downloadLink).toHaveAttribute(
    "href",
    `${JSON_DATA_URL_PREFIX}${encodeURIComponent(
      JSON.stringify(datumSet, null, 2),
    )}`,
  );
});
