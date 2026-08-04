import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import CasinoIcon from "@mui/icons-material/Casino";
import RefreshIcon from "@mui/icons-material/Refresh";
import {
  Box,
  FormControlLabel,
  IconButton,
  Snackbar,
  Switch,
  TextField,
  Tooltip,
} from "@mui/material";
import { useState } from "react";

import { copyTextToClipboard } from "../../nonview/base/Clipboard.js";
import RecentVisualQueries from "../../nonview/base/RecentVisualQueries.js";
import {
  SHARE_LINK_FEEDBACK_DURATION_MS,
  VISUAL_CONTENT_MAX_WIDTH_PX,
} from "../../nonview/constants/APP.js";
import { EXAMPLE_QUERIES } from "../../nonview/constants/ExampleQueries.js";
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
  const randomQueries = Array.from(
    new Set([
      ...RecentVisualQueries.read().map(({ query }) => query),
      ...EXAMPLE_QUERIES.map(({ query }) => query),
    ]),
  );

  function submit(event) {
    event?.preventDefault();
    onSubmit();
  }

  function submitExpertQuery(event) {
    if (event.key === "Enter") {
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

  function selectRandomQuery() {
    const randomIndex = Math.floor(Math.random() * randomQueries.length);
    const randomQuery = randomQueries[randomIndex];
    onChange(randomQuery);
    onSubmit(randomQuery);
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
        <TextField
          multiline
          minRows={2}
          maxRows={6}
          size="small"
          label="Visual query"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={submitExpertQuery}
          helperText="Press Enter to update"
          slotProps={{
            htmlInput: {
              autoComplete: "off",
              spellCheck: false,
            },
          }}
          sx={{
            width: "100%",
            "& .MuiInputBase-input": {
              overflowWrap: "anywhere",
            },
          }}
        />
        <Box
          sx={{
            alignItems: "center",
            display: "flex",
            justifyContent: "flex-end",
            mt: 1,
          }}
        >
          <Box sx={{ display: "flex" }}>
            <Tooltip title="Choose a random query">
              <span>
                <IconButton
                  aria-label="Choose a random query"
                  disabled={randomQueries.length === 0}
                  onClick={selectRandomQuery}
                  type="button"
                >
                  <CasinoIcon />
                </IconButton>
              </span>
            </Tooltip>
            <Tooltip title="Copy share link">
              <IconButton
                aria-label="Copy share link"
                onClick={copyShareLink}
                type="button"
              >
                <ContentCopyIcon />
              </IconButton>
            </Tooltip>
            <Tooltip title="Update visualization">
              <IconButton
                aria-label="Update visualization"
                color="primary"
                disabled={
                  loadedVisualQuery !== null &&
                  value.trim() === loadedVisualQuery
                }
                type="submit"
              >
                <RefreshIcon />
              </IconButton>
            </Tooltip>
          </Box>
          <FormControlLabel
            control={
              <Switch
                checked={editorOpen}
                onChange={(event) => setEditorOpen(event.target.checked)}
              />
            }
            label="Editor"
          />
        </Box>
        {editorOpen && (
          <LaypersonVisualQueryInput
            value={value}
            onChange={onChange}
            onSubmit={submit}
            queryOptions={queryOptions}
          />
        )}
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
