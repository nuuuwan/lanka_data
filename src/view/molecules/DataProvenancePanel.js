import { Box, Link, Paper, Typography } from "@mui/material";

import styles from "./DataProvenancePanel.module.css";

export default function DataProvenancePanel({ provenance }) {
  if (!provenance?.length) {
    return null;
  }

  return (
    <Paper className={styles.root} component="aside" variant="outlined">
      <Typography component="h2" variant="subtitle2">
        About this data
      </Typography>
      {provenance.map(({ source, url }, index) => (
        <Box className={styles.entry} key={`${source}-${index}`}>
          {source && (
            <Typography variant="body2">
              <strong>Source:</strong>{" "}
              {url ? (
                <Link href={url} target="_blank" rel="noopener noreferrer">
                  {source}
                </Link>
              ) : (
                source
              )}
            </Typography>
          )}
        </Box>
      ))}
    </Paper>
  );
}
