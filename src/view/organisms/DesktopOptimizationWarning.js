import { useEffect, useState } from "react";
import { Alert, Snackbar, useMediaQuery } from "@mui/material";

import { MOBILE_WARNING_STORAGE_KEY } from "../../nonview/constants/APP.js";
import styles from "./DesktopOptimizationWarning.module.css";

function hasSeenWarning() {
  try {
    return sessionStorage.getItem(MOBILE_WARNING_STORAGE_KEY) === "true";
  } catch (error) {
    console.warn(
      "[DesktopOptimizationWarning] Could not read session storage",
      error,
    );
    return false;
  }
}

export default function DesktopOptimizationWarning() {
  const isSmallScreen = useMediaQuery((theme) => theme.breakpoints.down("sm"));
  const [hasShown, setHasShown] = useState(hasSeenWarning);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!isSmallScreen || hasShown) return;
    setOpen(true);
    setHasShown(true);
    try {
      sessionStorage.setItem(MOBILE_WARNING_STORAGE_KEY, "true");
    } catch (error) {
      console.warn(
        "[DesktopOptimizationWarning] Could not save warning status",
        error,
      );
    }
  }, [hasShown, isSmallScreen]);

  return (
    <Snackbar
      open={open && isSmallScreen}
      anchorOrigin={{ vertical: "top", horizontal: "center" }}
      className={styles.notice}
      onClose={(_, reason) => {
        if (reason !== "clickaway") setOpen(false);
      }}
    >
      <Alert severity="warning" onClose={() => setOpen(false)}>
        Lanka Data is optimized for desktop, though it also works on mobile.
      </Alert>
    </Snackbar>
  );
}
