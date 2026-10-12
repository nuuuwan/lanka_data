import { Box, Button } from "@mui/material";

import { QUERY_WIZARD_STEPS } from "../../nonview/constants/QueryWizard.js";

export default function QueryWizardNavigation({
  step,
  onBack,
  onClose,
  refreshDisabled,
}) {
  return (
    <Box sx={{ display: "flex", justifyContent: "flex-end", gap: 1, mt: 2 }}>
      <Button onClick={onClose} type="button">
        Close
      </Button>
      <Button disabled={step === 0} onClick={onBack} type="button">
        Back
      </Button>
      {step < QUERY_WIZARD_STEPS.length - 1 ? (
        <Button type="submit" variant="contained">
          Next
        </Button>
      ) : (
        <Button disabled={refreshDisabled} type="submit" variant="contained">
          Update visualization
        </Button>
      )}
    </Box>
  );
}
