import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Button,
  Grid,
  Typography,
} from "@mui/material";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { EXAMPLE_QUERIES } from "../../nonview/constants/ExampleQueries.js";

export default function ExampleQueryGallery() {
  const [isExpanded, setIsExpanded] = useState(false);
  const navigate = useNavigate();

  return (
    <Accordion
      expanded={isExpanded}
      onChange={(_event, expanded) => setIsExpanded(expanded)}
      sx={{ mb: 2 }}
    >
      <AccordionSummary
        expandIcon={<ExpandMoreIcon />}
        aria-controls="example-query-gallery-content"
        id="example-query-gallery-header"
      >
        <Typography component="h2" variant="h6">
          Example Queries
        </Typography>
      </AccordionSummary>
      <AccordionDetails id="example-query-gallery-content">
        <Grid container spacing={1}>
          {EXAMPLE_QUERIES.map(({ label, description, query }) => (
            <Grid key={query} size={{ xs: 12, sm: 6 }}>
              <Button
                fullWidth
                variant="outlined"
                onClick={() => navigate(`/${query}`)}
                sx={{
                  alignItems: "flex-start",
                  flexDirection: "column",
                  height: "100%",
                  textAlign: "left",
                }}
              >
                <Typography component="span" variant="subtitle2">
                  {label}
                </Typography>
                <Typography
                  component="span"
                  variant="caption"
                  sx={{ color: "text.secondary", textTransform: "none" }}
                >
                  {description}
                </Typography>
              </Button>
            </Grid>
          ))}
        </Grid>
      </AccordionDetails>
    </Accordion>
  );
}
