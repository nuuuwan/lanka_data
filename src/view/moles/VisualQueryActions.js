import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import CasinoIcon from "@mui/icons-material/Casino";
import EditIcon from "@mui/icons-material/Edit";
import RefreshIcon from "@mui/icons-material/Refresh";
import { Box, IconButton, Tooltip, Typography } from "@mui/material";

export default function VisualQueryActions({
  disabled,
  randomDisabled,
  refreshDisabled,
  isEdited,
  onRandom,
  onCopy,
  onOpenEditor,
}) {
  return (
    <Box
      sx={{ alignItems: "center", display: "flex", justifyContent: "flex-end" }}
    >
      {isEdited && <Typography variant="caption">Edited</Typography>}
      <Tooltip title="Choose a random query">
        <span>
          <IconButton
            aria-label="Choose a random query"
            disabled={disabled || randomDisabled}
            onClick={onRandom}
            type="button"
          >
            <CasinoIcon />
          </IconButton>
        </span>
      </Tooltip>
      <Tooltip title="Copy share link">
        <span>
          <IconButton
            aria-label="Copy share link"
            disabled={disabled}
            onClick={onCopy}
            type="button"
          >
            <ContentCopyIcon />
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
      <Tooltip title="Query Wizard">
        <span>
          <IconButton
            aria-label="Query Wizard"
            aria-haspopup="dialog"
            disabled={disabled}
            onClick={onOpenEditor}
            type="button"
          >
            <EditIcon />
          </IconButton>
        </span>
      </Tooltip>
    </Box>
  );
}
