import CloseIcon from "@mui/icons-material/Close";
import { Box, Drawer, IconButton, Tooltip, Typography } from "@mui/material";

import LaypersonVisualQueryInput from "./LaypersonVisualQueryInput.js";
import styles from "./QueryWizardDrawer.module.css";

export default function QueryWizardDrawer({
  open,
  barHeight,
  onClose,
  disabled,
  value,
  onChange,
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
          "aria-labelledby": "query-wizard-title",
        },
      }}
    >
      <Box className={styles.content}>
        <Box className={styles.heading}>
          <Typography id="query-wizard-title" component="h2" variant="h6">
            Query Wizard
          </Typography>
          <Tooltip title="Close Query Wizard">
            <IconButton aria-label="Close Query Wizard" onClick={onClose}>
              <CloseIcon />
            </IconButton>
          </Tooltip>
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
          Close the wizard, then use Update visualization or Enter in the query
          bar.
        </Typography>
      </Box>
    </Drawer>
  );
}
