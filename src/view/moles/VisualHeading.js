import { Box, Typography } from "@mui/material";

import VisualMetadata from "../../nonview/core/VisualMetadata.js";
import styles from "./VisualHeading.module.css";

export default function VisualHeading({ query, datumSet }) {
  const { title, subtitle } = VisualMetadata.from(query, datumSet);

  return (
    <Box component="header" className={styles.root}>
      <Typography component="h1" variant="h4">
        {title}
      </Typography>
      <Typography color="text.secondary" variant="body1">
        {subtitle}
      </Typography>
    </Box>
  );
}
