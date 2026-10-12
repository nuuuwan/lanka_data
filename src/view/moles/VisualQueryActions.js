import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import AutoFixHighIcon from "@mui/icons-material/AutoFixHigh";
import RefreshIcon from "@mui/icons-material/Refresh";
import { Box, IconButton, Tooltip, Typography } from "@mui/material";

export default function VisualQueryActions({
  disabled,
  refreshDisabled,
  copyDisabled = false,
  isEdited,
  onCopy,
  onOpenEditor,
  children,
}) {
  return (
    <Box
      sx={{ alignItems: "center", display: "flex", justifyContent: "flex-end" }}
    >
      {isEdited && <Typography variant="caption">Edited</Typography>}
      <Tooltip title="Query Wizard">
        <span>
          <IconButton
            aria-label="Query Wizard"
            aria-haspopup="dialog"
            disabled={disabled}
            onClick={onOpenEditor}
            type="button"
          >
            <AutoFixHighIcon />
          </IconButton>
        </span>
      </Tooltip>
      <Tooltip title="Update visualization">
        <span>
          <IconButton
            aria-label="Update visualization"
            color="primary"
            disabled={disabled || refreshDisabled}
            type="submit"
          >
            <RefreshIcon />
          </IconButton>
        </span>
      </Tooltip>
      <Tooltip title="Copy share link">
        <span>
          <IconButton
            aria-label="Copy share link"
            disabled={disabled || copyDisabled}
            onClick={onCopy}
            type="button"
          >
            <ContentCopyIcon />
          </IconButton>
        </span>
      </Tooltip>
      {children}
    </Box>
  );
}
