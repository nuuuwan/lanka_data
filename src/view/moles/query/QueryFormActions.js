import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import RefreshIcon from "@mui/icons-material/Refresh";
import { Box, Button } from "@mui/material";

export default function QueryFormActions({ disableUpdate, onCopyShareLink }) {
  return (
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
        onClick={onCopyShareLink}
        startIcon={<ContentCopyIcon />}
      >
        Copy Share Link
      </Button>
      <Button
        type="submit"
        variant="contained"
        startIcon={<RefreshIcon />}
        disabled={disableUpdate}
      >
        Update
      </Button>
    </Box>
  );
}
