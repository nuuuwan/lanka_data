import { Box, Paper, Snackbar } from "@mui/material";
import { useRef, useState } from "react";

import useElementHeight from "../../nonview/base/useElementHeight.js";
import useVisualQueryActions from "../../nonview/core/useVisualQueryActions.js";
import { SHARE_LINK_FEEDBACK_DURATION_MS } from "../../nonview/constants/APP.js";
import VisualQueryActions from "../moles/VisualQueryActions.js";
import QueryWizardDrawer from "./QueryWizardDrawer.js";
import VisualQueryTextInput from "../moles/VisualQueryTextInput.js";
import styles from "./VisualQueryForm.module.css";

export default function VisualQueryForm({
  disabled = false,
  value,
  onChange,
  onSubmit,
  queryOptions,
  loadedVisualQuery = null,
}) {
  const [editorOpen, setEditorOpen] = useState(false);
  const barRef = useRef(null);
  const barHeight = useElementHeight(barRef);
  const actions = useVisualQueryActions(queryOptions, onChange, onSubmit);
  const isEdited =
    loadedVisualQuery !== null && value.trim() !== loadedVisualQuery;
  const refreshDisabled = loadedVisualQuery !== null && !isEdited;
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
            disabled={disabled}
            aria-busy={disabled}
            className={styles.fields}
          >
            <VisualQueryTextInput
              compact
              disabled={disabled}
              onChange={onChange}
              value={value}
              onKeyDown={(event) => {
                if (event.key === "Enter") submit(event);
              }}
            />
            <VisualQueryActions
              disabled={disabled}
              randomDisabled={actions.randomDisabled}
              refreshDisabled={refreshDisabled}
              isEdited={isEdited}
              onRandom={actions.selectRandomQuery}
              onCopy={actions.copyShareLink}
              onOpenEditor={() => setEditorOpen(true)}
            />
          </Box>
        </Box>
      </Paper>
      <QueryWizardDrawer
        barHeight={barHeight}
        open={editorOpen}
        onClose={() => setEditorOpen(false)}
        disabled={disabled}
        refreshDisabled={refreshDisabled}
        value={value}
        onChange={onChange}
        onSubmit={submit}
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
