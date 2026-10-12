import CloseIcon from "@mui/icons-material/Close";
import { Box, Button, Drawer, IconButton, Typography } from "@mui/material";

import LaypersonVisualQueryInput from "./LaypersonVisualQueryInput.js";
import VisualQueryEditorHelp from "./VisualQueryEditorHelp.js";
import styles from "./VisualQueryEditorDrawer.module.css";

export default function VisualQueryEditorDrawer({
  open,
  barHeight,
  onClose,
  disabled,
  refreshDisabled,
  value,
  onChange,
  onSubmit,
  queryOptions,
}) {
  return (
    <Drawer
      anchor="bottom"
      open={open}
      onClose={onClose}
      slotProps={{
        backdrop: { sx: { bottom: barHeight } },
        paper: {
          className: styles.drawer,
          style: { "--query-bar-height": `${barHeight}px` },
          role: "dialog",
          "aria-modal": true,
          "aria-labelledby": "visual-editor-title",
        },
      }}
    >
      <Box className={styles.content}>
        <Box className={styles.heading}>
          <Typography id="visual-editor-title" component="h2" variant="h6">
            Visual editor
          </Typography>
          <IconButton aria-label="Close visual editor" onClick={onClose}>
            <CloseIcon />
          </IconButton>
        </Box>
        <VisualQueryEditorHelp />
        <Box component="form" onSubmit={onSubmit}>
          <Box
            component="fieldset"
            disabled={disabled}
            className={styles.fields}
          >
            <LaypersonVisualQueryInput
              value={value}
              onChange={onChange}
              onSubmit={onSubmit}
              queryOptions={queryOptions}
            />
            <Box className={styles.actions}>
              <Button onClick={onClose} type="button">
                Close
              </Button>
              <Button
                disabled={refreshDisabled}
                type="submit"
                variant="contained"
              >
                Update visualization
              </Button>
            </Box>
          </Box>
        </Box>
      </Box>
    </Drawer>
  );
}
