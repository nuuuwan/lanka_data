import { Box, Snackbar } from "@mui/material";
import { useState } from "react";

import { copyTextToClipboard } from "../../nonview/base/Clipboard.js";
import { SHARE_LINK_FEEDBACK_DURATION_MS } from "../../nonview/constants/APP.js";
import VisualQueryActions from "../moles/VisualQueryActions.js";
import VisualQueryFields from "../moles/VisualQueryFields.js";

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

  function submitExpertQuery(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
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
        <VisualQueryFields
          mode={mode}
          onModeChange={setMode}
          onChange={onChange}
          onExpertKeyDown={submitExpertQuery}
          onSubmit={submit}
          queryOptions={queryOptions}
          value={value}
        />
      </Box>
      <VisualQueryActions
        loadedVisualQuery={loadedVisualQuery}
        onCopy={copyShareLink}
        value={value}
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
