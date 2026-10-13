import CheckIcon from "@mui/icons-material/Check";
import CloseIcon from "@mui/icons-material/Close";
import RestartAltIcon from "@mui/icons-material/RestartAlt";
import { Box, Drawer, IconButton, Tooltip } from "@mui/material";

import LaypersonVisualQueryInput from "./LaypersonVisualQueryInput.js";
import styles from "./QueryWizardDrawer.module.css";

export default function QueryWizardDrawer({
  open,
  barHeight,
  onClose,
  onSubmit,
  onReset,
  resetDisabled = false,
  isEdited = false,
  disabled,
  value,
  onChange,
  queryOptions,
}) {
  const handleCloseAndUpdate = () => {
    onSubmit?.();
  };

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
          "aria-label": "Query Wizard",
        },
      }}
    >
      <Box className={styles.content}>
        <Box component="form" onSubmit={(event) => event.preventDefault()}>
          <Box
            component="fieldset"
            disabled={disabled}
            className={styles.fields}
          >
            <LaypersonVisualQueryInput
              value={value}
              onChange={onChange}
              queryOptions={queryOptions}
            />
          </Box>
        </Box>
        <Box className={styles.actions}>
          <Tooltip title="Reset to active query">
            <span>
              <IconButton
                aria-label="Reset to active query"
                disabled={disabled || resetDisabled}
                onClick={onReset}
                size="small"
              >
                <RestartAltIcon />
              </IconButton>
            </span>
          </Tooltip>
          {isEdited ? (
            <Tooltip title="Close and update">
              <span>
                <IconButton
                  aria-label="Close and update"
                  color="primary"
                  disabled={disabled || !value?.trim()}
                  onClick={handleCloseAndUpdate}
                >
                  <CheckIcon />
                </IconButton>
              </span>
            </Tooltip>
          ) : (
            <Tooltip title="Close">
              <span>
                <IconButton
                  aria-label="Close"
                  disabled={disabled}
                  onClick={onClose}
                >
                  <CloseIcon />
                </IconButton>
              </span>
            </Tooltip>
          )}
        </Box>
      </Box>
    </Drawer>
  );
}
