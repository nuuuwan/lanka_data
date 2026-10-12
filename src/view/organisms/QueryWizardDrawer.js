import { useEffect, useState } from "react";
import CloseIcon from "@mui/icons-material/Close";
import { Box, Drawer, IconButton, Typography } from "@mui/material";

import { QUERY_WIZARD_STEPS } from "../../nonview/constants/QueryWizard.js";
import LaypersonVisualQueryInput from "../moles/LaypersonVisualQueryInput.js";
import QueryWizardNavigation from "../moles/QueryWizardNavigation.js";
import QueryWizardReview from "../moles/QueryWizardReview.js";
import QueryWizardProgress from "../moles/QueryWizardProgress.js";
import styles from "./QueryWizardDrawer.module.css";

export default function QueryWizardDrawer({
  open,
  barHeight,
  onClose,
  disabled,
  refreshDisabled,
  value,
  onChange,
  onSubmit,
  queryOptions,
}) {
  const [step, setStep] = useState(0);
  useEffect(() => {
    if (open) setStep(0);
  }, [open]);
  const isReview = step === QUERY_WIZARD_STEPS.length - 1;
  const advance = (event) => {
    event?.preventDefault();
    if (isReview) onSubmit();
    else setStep((current) => current + 1);
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
          <IconButton aria-label="Close Query Wizard" onClick={onClose}>
            <CloseIcon />
          </IconButton>
        </Box>
        <QueryWizardProgress step={step} />
        <Box component="form" onSubmit={advance}>
          <Box
            component="fieldset"
            disabled={disabled}
            className={styles.fields}
          >
            <Typography
              component="h3"
              variant="subtitle1"
              sx={{ mb: 2 }}
              aria-live="polite"
            >
              Step {step + 1}: {QUERY_WIZARD_STEPS[step]}
            </Typography>
            {isReview ? (
              <QueryWizardReview value={value} />
            ) : (
              <LaypersonVisualQueryInput
                step={QUERY_WIZARD_STEPS[step]}
                value={value}
                onChange={onChange}
                onSubmit={advance}
                queryOptions={queryOptions}
              />
            )}
            <QueryWizardNavigation
              step={step}
              onBack={() => setStep((current) => current - 1)}
              onClose={onClose}
              refreshDisabled={refreshDisabled}
            />
          </Box>
        </Box>
      </Box>
    </Drawer>
  );
}
