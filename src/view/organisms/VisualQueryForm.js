import { Box, Snackbar } from "@mui/material";
import { useState } from "react";

import { copyTextToClipboard } from "../../nonview/base/Clipboard.js";
import { SHARE_LINK_FEEDBACK_DURATION_MS } from "../../nonview/constants/APP.js";
import ExpertQueryInput from "../moles/query/ExpertQueryInput.js";
import LaypersonVisualQueryInput from "../moles/query/LaypersonVisualQueryInput.js";
import QueryFormActions from "../moles/query/QueryFormActions.js";
import QueryModeTabs from "../moles/query/QueryModeTabs.js";

export default function VisualQueryForm({
  disabled = false,
  value,
  onChange,
  onSubmit,
  queryOptions,
  loadedVisualQuery = null,
}) {
  const [mode, setMode] = useState("expert");
  const [shareFeedback, setShareFeedback] = useState(null);

  function submit(event) {
    event?.preventDefault();
    onSubmit();
  }

  async function copyShareLink() {
    try {
      await copyTextToClipboard(window.location.href);
      setShareFeedback("Share link copied");
    } catch {
      setShareFeedback("Could not copy share link");
    }
  }

  return (
    <Box
      component="form"
      aria-label="Visual query form"
      onSubmit={submit}
      sx={{ mb: 2 }}
    >
      <Box
        component="fieldset"
        disabled={disabled}
        aria-busy={disabled}
        sx={{ border: 0, m: 0, minWidth: 0, p: 0 }}
      >
        <QueryModeTabs mode={mode} onChange={setMode} />
        {mode === "layperson" ? (
          <LaypersonVisualQueryInput
            value={value}
            onChange={onChange}
            onSubmit={submit}
            queryOptions={queryOptions}
          />
        ) : (
          <ExpertQueryInput
            value={value}
            onChange={onChange}
            onSubmit={submit}
          />
        )}
      </Box>
      <QueryFormActions
        disableUpdate={
          loadedVisualQuery !== null && value.trim() === loadedVisualQuery
        }
        onCopyShareLink={copyShareLink}
      />
      <Snackbar
        open={shareFeedback !== null}
        autoHideDuration={SHARE_LINK_FEEDBACK_DURATION_MS}
        message={shareFeedback}
        onClose={() => setShareFeedback(null)}
      />
    </Box>
  );
}
