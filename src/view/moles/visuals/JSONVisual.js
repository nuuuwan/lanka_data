import { Button } from "@mui/material";

import {
  JSON_DATA_URL_PREFIX,
  JSON_DOWNLOAD_FILE_NAME,
} from "../../../nonview/constants/APP.js";

export default function JSONVisual({ datumSet }) {
  const json = JSON.stringify(datumSet, null, 2);
  const downloadURL = `${JSON_DATA_URL_PREFIX}${encodeURIComponent(json)}`;

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
