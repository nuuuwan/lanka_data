import { Button } from "@mui/material";

import { getJSONDownloadURL } from "../../../nonview/base/RawJSON.js";
import { JSON_DOWNLOAD_FILE_NAME } from "../../../nonview/constants/APP.js";

export default function JSONVisual({ datumSet }) {
  const json = JSON.stringify(datumSet, null, 2);
  const downloadURL = getJSONDownloadURL(window.location, json);

  return (
    <>
      <pre>{json}</pre>
      <Button
        component="a"
        download={JSON_DOWNLOAD_FILE_NAME}
        href={downloadURL}
        variant="contained"
      >
        Download JSON
      </Button>
    </>
  );
}
