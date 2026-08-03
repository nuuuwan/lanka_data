import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import RefreshIcon from "@mui/icons-material/Refresh";
import {
  Box,
  Button,
  Snackbar,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
} from "@mui/material";
import { useState } from "react";

import { copyTextToClipboard } from "../../nonview/base/Clipboard.js";
import { SHARE_LINK_FEEDBACK_DURATION_MS } from "../../nonview/constants/APP.js";
import LaypersonVisualQueryInput from "../moles/LaypersonVisualQueryInput.js";

export default function VisualQueryForm({
  value,
  onChange,
  onSubmit,
  queryOptions,
}) {
  const [mode, setMode] = useState("layperson");
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
      <ToggleButtonGroup
        exclusive
        size="small"
        value={mode}
        onChange={(_event, nextMode) => nextMode && setMode(nextMode)}
        aria-label="Query input mode"
        sx={{ mb: 1.5 }}
      >
        <ToggleButton value="layperson">Layperson Mode</ToggleButton>
        <ToggleButton value="expert">Expert Mode</ToggleButton>
      </ToggleButtonGroup>
      {mode === "layperson" ? (
        <LaypersonVisualQueryInput
          value={value}
          onChange={onChange}
          onSubmit={submit}
          queryOptions={queryOptions}
        />
      ) : (
        <TextField
          fullWidth
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
            "& .MuiInputBase-input": {
              fontFamily: "monospace",
              overflowWrap: "anywhere",
            },
          }}
        />
      )}
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
