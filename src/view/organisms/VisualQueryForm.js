import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import RefreshIcon from "@mui/icons-material/Refresh";
import {
  Box,
  Button,
  FormControlLabel,
  Snackbar,
  Switch,
  TextField,
} from "@mui/material";
import { useState } from "react";

import { copyTextToClipboard } from "../../nonview/base/Clipboard.js";
import {
  SHARE_LINK_FEEDBACK_DURATION_MS,
  VISUAL_CONTENT_MAX_WIDTH_PX,
} from "../../nonview/constants/APP.js";
import LaypersonVisualQueryInput from "../moles/LaypersonVisualQueryInput.js";

export default function VisualQueryForm({
  disabled = false,
  value,
  onChange,
  onSubmit,
  queryOptions,
  loadedVisualQuery = null,
}) {
  const [editorOpen, setEditorOpen] = useState(false);
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
      sx={{ mb: 2, maxWidth: VISUAL_CONTENT_MAX_WIDTH_PX, mx: "auto" }}
    >
      <Box
        component="fieldset"
        disabled={disabled}
        aria-busy={disabled}
        sx={{ border: 0, m: 0, minWidth: 0, p: 0 }}
      >
        <FormControlLabel
          control={
            <Switch
              checked={editorOpen}
              onChange={(event) => setEditorOpen(event.target.checked)}
            />
          }
          label="Editor"
          sx={{ mb: 1 }}
        />
        {editorOpen ? (
          <LaypersonVisualQueryInput
            value={value}
            onChange={onChange}
            onSubmit={submit}
            queryOptions={queryOptions}
          />
        ) : (
          <TextField
            multiline
            minRows={2}
            maxRows={6}
            size="small"
            label="Visual query"
            value={value}
            onChange={(event) => onChange(event.target.value)}
            onKeyDown={submitExpertQuery}
            helperText="Press Enter to update; use Shift+Enter for a new line"
            slotProps={{
              htmlInput: {
                autoComplete: "off",
                spellCheck: false,
              },
            }}
            sx={{
              maxWidth: VISUAL_CONTENT_MAX_WIDTH_PX,
              width: "100%",
              "& .MuiInputBase-input": {
                overflowWrap: "anywhere",
              },
            }}
          />
        )}
      </Box>
      <Box
        sx={{
          display: "flex",
          justifyContent: "flex-end",
          gap: 1,
          mt: 1.5,
        }}
      >
        <Button
          type="button"
          variant="outlined"
          onClick={copyShareLink}
          startIcon={<ContentCopyIcon />}
        >
          Copy Share Link
        </Button>
        <Button
          type="submit"
          variant="contained"
          startIcon={<RefreshIcon />}
          disabled={
            loadedVisualQuery !== null && value.trim() === loadedVisualQuery
          }
        >
          Update
        </Button>
      </Box>
      <Snackbar
        open={shareFeedback !== null}
        autoHideDuration={SHARE_LINK_FEEDBACK_DURATION_MS}
        message={shareFeedback}
        onClose={() => setShareFeedback(null)}
      />
    </Box>
  );
}
