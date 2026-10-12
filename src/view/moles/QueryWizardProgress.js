import { Alert, Step, StepLabel, Stepper } from "@mui/material";

import {
  QUERY_WIZARD_HELP,
  QUERY_WIZARD_STEPS,
} from "../../nonview/constants/QueryWizard.js";

export default function QueryWizardProgress({ step }) {
  return (
    <>
      <Stepper activeStep={step} alternativeLabel sx={{ mb: 2 }}>
        {QUERY_WIZARD_STEPS.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>
      <Alert severity="info" sx={{ mb: 2 }}>
        {QUERY_WIZARD_HELP[step]}
      </Alert>
    </>
  );
}
