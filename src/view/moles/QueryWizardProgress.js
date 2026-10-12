import { Alert, Step, StepButton, Stepper } from "@mui/material";

import {
  QUERY_WIZARD_HELP,
  QUERY_WIZARD_STEPS,
} from "../../nonview/constants/QueryWizard.js";
import styles from "./QueryWizardProgress.module.css";

export default function QueryWizardProgress({
  step,
  onStepChange,
  disabled,
  helpVisible,
  onCloseHelp,
}) {
  const help = QUERY_WIZARD_HELP[step];
  return (
    <>
      <Stepper nonLinear activeStep={step} alternativeLabel sx={{ mb: 2 }}>
        {QUERY_WIZARD_STEPS.map((label, index) => (
          <Step key={label}>
            <StepButton
              disabled={disabled}
              onClick={() => onStepChange(index)}
              aria-current={step === index ? "step" : undefined}
            >
              {label}
            </StepButton>
          </Step>
        ))}
      </Stepper>
      {helpVisible && (
        <Alert severity="info" sx={{ mb: 2 }} onClose={onCloseHelp}>
          <p className={styles.paragraph}>{help.introduction}</p>
          <ul className={styles.examples}>
            {help.examples.map((example) => (
              <li key={example}>{example}</li>
            ))}
          </ul>
          {help.tip && <p className={styles.paragraph}>{help.tip}</p>}
        </Alert>
      )}
    </>
  );
}
