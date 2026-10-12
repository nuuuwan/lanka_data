import { useState } from "react";
import DownloadIcon from "@mui/icons-material/Download";
import { Alert, IconButton, Snackbar, Tooltip } from "@mui/material";

import downloadElementPNG from "../../nonview/base/downloadElementPNG.js";
import fileNameFromURL from "../../nonview/base/fileNameFromURL.js";
import {
  VISUAL_PNG_PIXEL_RATIO,
  SHARE_LINK_FEEDBACK_DURATION_MS,
} from "../../nonview/constants/APP.js";

export default function DownloadVisualButton({
  visualElement,
  bottomOffset,
  disabled = false,
}) {
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState(null);
  async function download() {
    setDownloading(true);
    setError(null);
    try {
      await downloadElementPNG(
        visualElement,
        fileNameFromURL(window.location.href, "png"),
        VISUAL_PNG_PIXEL_RATIO,
      );
    } catch (cause) {
      console.error("[DownloadVisualButton] PNG download failed", cause);
      setError("Could not download the visual. Please try again.");
    } finally {
      setDownloading(false);
    }
  }
  return (
    <>
      <Tooltip title="Download visual as PNG">
        <span>
          <IconButton
            aria-label="Download visual as PNG"
            disabled={disabled || downloading || !visualElement}
            type="button"
            aria-busy={downloading}
            onClick={download}
          >
            <DownloadIcon />
          </IconButton>
        </span>
      </Tooltip>
      <Snackbar
        open={Boolean(error)}
        onClose={() => setError(null)}
        autoHideDuration={SHARE_LINK_FEEDBACK_DURATION_MS}
        sx={{ bottom: `${bottomOffset}px !important` }}
      >
        <Alert severity="error" onClose={() => setError(null)}>
          {error}
        </Alert>
      </Snackbar>
    </>
  );
}
