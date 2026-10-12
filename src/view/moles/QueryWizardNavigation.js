import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import ArrowForwardIcon from "@mui/icons-material/ArrowForward";
import { Box, IconButton, Tooltip } from "@mui/material";

import { QUERY_WIZARD_STEPS } from "../../nonview/constants/QueryWizard.js";

export default function QueryWizardNavigation({
  step,
  onBack,
  onNext,
  disabled,
}) {
  return (
    <Box sx={{ display: "flex", justifyContent: "flex-end", gap: 1 }}>
      <Tooltip title="Back">
        <span>
          <IconButton
            aria-label="Back"
            disabled={disabled || step === 0}
            onClick={onBack}
            type="button"
          >
            <ArrowBackIcon />
          </IconButton>
        </span>
      </Tooltip>
      <Tooltip title="Next">
        <span>
          <IconButton
            aria-label="Next"
            disabled={disabled || step === QUERY_WIZARD_STEPS.length - 1}
            onClick={onNext}
            type="button"
            color="primary"
          >
            <ArrowForwardIcon />
          </IconButton>
        </span>
      </Tooltip>
    </Box>
  );
}
