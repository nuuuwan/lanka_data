import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import ArrowForwardIcon from "@mui/icons-material/ArrowForward";
import RefreshIcon from "@mui/icons-material/Refresh";
import { Box, IconButton, Tooltip } from "@mui/material";

import { QUERY_WIZARD_STEPS } from "../../nonview/constants/QueryWizard.js";

export default function QueryWizardNavigation({
  step,
  onBack,
  refreshDisabled,
}) {
  return (
    <Box sx={{ display: "flex", justifyContent: "flex-end", gap: 1, mt: 2 }}>
      <Tooltip title="Back">
        <span>
          <IconButton
            aria-label="Back"
            disabled={step === 0}
            onClick={onBack}
            type="button"
          >
            <ArrowBackIcon />
          </IconButton>
        </span>
      </Tooltip>
      {step < QUERY_WIZARD_STEPS.length - 1 ? (
        <Tooltip title="Next">
          <IconButton aria-label="Next" type="submit" color="primary">
            <ArrowForwardIcon />
          </IconButton>
        </Tooltip>
      ) : (
        <Tooltip title="Update visualization">
          <span>
            <IconButton
              aria-label="Update visualization"
              disabled={refreshDisabled}
              type="submit"
              color="primary"
            >
              <RefreshIcon />
            </IconButton>
          </span>
        </Tooltip>
      )}
    </Box>
  );
}
