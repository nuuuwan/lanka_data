import { CircularProgress, Typography } from "@mui/material";

import ProgressList from "./ProgressList.js";
import styles from "./ProgressList.module.css";

export default function LoadingProgress({ heading = "Loading visual", steps }) {
  return (
    <section className={styles.loading} aria-label="Loading visual">
      <div className={styles.heading}>
        <CircularProgress size="1rem" thickness={5} aria-label="Loading" />
        <Typography component="h2" variant="body1" className={styles.title}>
          {heading}
        </Typography>
      </div>
      <ProgressList steps={steps} />
    </section>
  );
}
