import { Box, Paper, Snackbar } from "@mui/material";
import { useRef, useState } from "react";
import { useParams } from "react-router-dom";

import useElementHeight from "../../nonview/base/useElementHeight.js";
import useVisualQueryActions from "../../nonview/core/useVisualQueryActions.js";
import { SHARE_LINK_FEEDBACK_DURATION_MS } from "../../nonview/constants/APP.js";
import VisualQueryActions from "../moles/VisualQueryActions.js";
import QueryWizardDrawer from "../moles/QueryWizardDrawer.js";
import DownloadVisualButton from "./DownloadVisualButton.js";
import VisualQueryTextInput from "../moles/VisualQueryTextInput.js";
import styles from "./VisualQueryForm.module.css";

export default function VisualQueryForm({
  disabled = false,
  value,
  onChange,
  onSubmit,
  queryOptions,
  loadedVisualQuery = null,
  visualElement = null,
}) {
  const { "*": routeQuery } = useParams();
  const currentQuery = routeQuery ?? loadedVisualQuery ?? "";
  const [editorOpen, setEditorOpen] = useState(false);
  const barDisabled = disabled || editorOpen;
  const barRef = useRef(null);
  const barHeight = useElementHeight(barRef);
  const actions = useVisualQueryActions();
  const query = value.trim();
  const isEdited = query !== currentQuery;
  const refreshDisabled = !query || !isEdited;
  const submit = (event) => {
    event?.preventDefault();
    onSubmit();
    setEditorOpen(false);
  };

  return (
    <>
      <Box aria-hidden="true" sx={{ height: barHeight }} />
      <Paper
        ref={barRef}
        component="section"
        aria-label="Visual query bar"
        elevation={0}
        className={styles.bar}
      >
        <Box component="form" aria-label="Visual query form" onSubmit={submit}>
          <Box
            component="fieldset"
            disabled={barDisabled}
            aria-busy={disabled}
            className={styles.fields}
          >
            <VisualQueryTextInput
              compact
              disabled={barDisabled}
              onChange={onChange}
              value={value}
              onKeyDown={(event) => {
                if (event.key === "Enter") submit(event);
              }}
            />
            <VisualQueryActions
              disabled={barDisabled}
              refreshDisabled={refreshDisabled}
              copyDisabled={!query}
              isEdited={isEdited}
              onCopy={actions.copyShareLink}
              onOpenEditor={() => setEditorOpen(true)}
            >
              <DownloadVisualButton
                disabled={barDisabled}
                visualElement={visualElement}
                bottomOffset={barHeight}
              />
            </VisualQueryActions>
          </Box>
        </Box>
      </Paper>
      <QueryWizardDrawer
        barHeight={barHeight}
        open={editorOpen}
        onClose={() => setEditorOpen(false)}
        onSubmit={submit}
        onReset={() => onChange(currentQuery)}
        resetDisabled={!isEdited}
        disabled={disabled}
        value={value}
        onChange={onChange}
        queryOptions={queryOptions}
      />
      <Snackbar
        open={actions.shareFeedback !== null}
        autoHideDuration={SHARE_LINK_FEEDBACK_DURATION_MS}
        message={actions.shareFeedback}
        onClose={actions.dismissShareFeedback}
        sx={{ bottom: `${barHeight}px !important` }}
      />
    </>
  );
}
