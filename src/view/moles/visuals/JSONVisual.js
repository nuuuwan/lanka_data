import DownloadIcon from "@mui/icons-material/Download";
import { Box, Button } from "@mui/material";

import { getJSONDownloadURL } from "../../../nonview/base/RawJSON.js";
import { JSON_DOWNLOAD_FILE_NAME } from "../../../nonview/constants/APP.js";

export default function JSONVisual({ datumSet }) {
  const json = JSON.stringify(datumSet, null, 2);
  const downloadURL = getJSONDownloadURL(window.location, json);

  return (
    <>
      <Box sx={{ display: "flex", justifyContent: "flex-end", mb: 1 }}>
        <Button
          component="a"
          download={JSON_DOWNLOAD_FILE_NAME}
          href={downloadURL}
          startIcon={<DownloadIcon />}
          variant="contained"
        >
          Download JSON
        </Button>
      </Box>
      <pre>{json}</pre>
    </>
  );
}
