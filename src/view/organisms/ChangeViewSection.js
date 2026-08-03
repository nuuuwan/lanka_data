import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Typography,
} from "@mui/material";
import { useState } from "react";

import VisualQueryForm from "./VisualQueryForm.js";

export default function ChangeViewSection({
  value,
  onChange,
  onSubmit,
  queryOptions,
}) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <Accordion
      expanded={isExpanded}
      onChange={(_event, expanded) => setIsExpanded(expanded)}
      sx={{ mt: 2 }}
    >
      <AccordionSummary
        expandIcon={<ExpandMoreIcon />}
        aria-controls="change-view-content"
        id="change-view-header"
      >
        <Typography component="h2" variant="h6">
          Change this view
        </Typography>
      </AccordionSummary>
      <AccordionDetails id="change-view-content">
        <VisualQueryForm
          value={value}
          onChange={onChange}
          onSubmit={onSubmit}
          queryOptions={queryOptions}
        />
      </AccordionDetails>
    </Accordion>
  );
}
