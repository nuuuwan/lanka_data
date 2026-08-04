import OpenInNewIcon from "@mui/icons-material/OpenInNew";
import { Box, Link, Typography } from "@mui/material";

import styles from "./DataProvenancePanel.module.css";

export default function DataProvenancePanel({ provenance }) {
  const sources = [
    ...new Map(
      provenance
        ?.filter(({ source }) => source)
        .map((item) => [[item.source, item.url].join("\n"), item]),
    ).values(),
  ];

  if (!sources.length) {
    return null;
  }

  return (
    <Box className={styles.root} component="aside">
      <Typography variant="body2">
        <strong>source:</strong>{" "}
        {sources.map(({ source, url }, index) => (
          <span key={`${source}-${index}`}>
            {index > 0 && ", "}
            {url ? (
              <Link
                href={url}
                target="_blank"
                rel="noopener noreferrer"
                sx={{
                  alignItems: "center",
                  display: "inline-flex",
                  gap: 0.25,
                }}
              >
                {source}
                <OpenInNewIcon sx={{ fontSize: "inherit" }} />
              </Link>
            ) : (
              source
            )}
          </span>
        ))}
      </Typography>
    </Box>
  );
}
