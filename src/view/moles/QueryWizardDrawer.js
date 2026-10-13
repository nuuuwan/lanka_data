import CheckIcon from "@mui/icons-material/Check";
import RestartAltIcon from "@mui/icons-material/RestartAlt";
import { Box, Drawer, IconButton, Tooltip, Typography } from "@mui/material";

import LaypersonVisualQueryInput from "./LaypersonVisualQueryInput.js";
import styles from "./QueryWizardDrawer.module.css";

export default function QueryWizardDrawer({
  open,
  barHeight,
  onClose,
  onSubmit,
  onReset,
  resetDisabled = false,
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
          "aria-labelledby": "query-wizard-title",
        },
      }}
    >
      <Box className={styles.content}>
        <Box className={styles.heading}>
          <Typography id="query-wizard-title" component="h2" variant="h6">
            Query Wizard
          </Typography>
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
            <Tooltip title="Close and update">
              <IconButton
                aria-label="Close and update"
                color="primary"
                disabled={disabled}
                onClick={handleCloseAndUpdate}
              >
                <CheckIcon />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>
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
        <Typography component="p" className={styles.tip}>
          Click the check button to close and update the visualization.
        </Typography>
      </Box>
    </Drawer>
  );
}
