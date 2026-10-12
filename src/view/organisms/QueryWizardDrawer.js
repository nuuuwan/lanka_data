import { useEffect, useState } from "react";
import CloseIcon from "@mui/icons-material/Close";
import { Box, Drawer, IconButton, Tooltip, Typography } from "@mui/material";

import { QUERY_WIZARD_STEPS } from "../../nonview/constants/QueryWizard.js";
import LaypersonVisualQueryInput from "../moles/LaypersonVisualQueryInput.js";
import QueryWizardNavigation from "../moles/QueryWizardNavigation.js";
import QueryWizardProgress from "../moles/QueryWizardProgress.js";
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
  const [step, setStep] = useState(0);
  const [hiddenHelpSteps, setHiddenHelpSteps] = useState([]);
  useEffect(() => {
    if (open) {
      setStep(0);
      setHiddenHelpSteps([]);
    }
  }, [open]);
  const advance = (event) => {
    event?.preventDefault();
    setStep((current) => Math.min(current + 1, QUERY_WIZARD_STEPS.length - 1));
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
          <QueryWizardNavigation
            step={step}
            onBack={() => setStep((current) => current - 1)}
            onNext={advance}
            disabled={disabled}
          />
        </Box>
        <QueryWizardProgress
          step={step}
          onStepChange={setStep}
          disabled={disabled}
          helpVisible={!hiddenHelpSteps.includes(step)}
          onCloseHelp={() =>
            setHiddenHelpSteps((current) => [...current, step])
          }
        />
        <Box component="form" onSubmit={advance}>
          <Box
            component="fieldset"
            disabled={disabled}
            className={styles.fields}
          >
            <LaypersonVisualQueryInput
              step={QUERY_WIZARD_STEPS[step]}
              value={value}
              onChange={onChange}
              onSubmit={advance}
              queryOptions={queryOptions}
            />
          </Box>
          <Box sx={{ display: "flex", justifyContent: "flex-end", mt: 2 }}>
            <Tooltip title="Close Query Wizard">
              <IconButton aria-label="Close Query Wizard" onClick={onClose}>
                <CloseIcon />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>
      </Box>
    </Drawer>
  );
}
