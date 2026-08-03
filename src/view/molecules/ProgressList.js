import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import RadioButtonUncheckedIcon from "@mui/icons-material/RadioButtonUnchecked";
import {
  CircularProgress,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
} from "@mui/material";

import FormatUtils from "../moles/visual_utils/FormatUtils.js";
import styles from "./ProgressList.module.css";

function getStatusText(step) {
  if (step.status === "complete" || step.status === "active") {
    return FormatUtils.humanizeDuration(step.durationSeconds);
  }
  return "Waiting";
}

function StatusIcon({ status }) {
  if (status === "complete") {
    return <CheckCircleIcon color="success" aria-label="Complete" />;
  }
  if (status === "active") {
    return <CircularProgress size="1.5rem" aria-label="Loading" />;
  }
  return <RadioButtonUncheckedIcon color="disabled" aria-label="Waiting" />;
}

export default function ProgressList({ steps }) {
  return (
    <List className={styles.root} aria-label="Visual loading progress">
      {steps.map((step) => (
        <ListItem key={step.label}>
          <ListItemIcon>
            <StatusIcon status={step.status} />
          </ListItemIcon>
          <ListItemText primary={step.label} secondary={getStatusText(step)} />
        </ListItem>
      ))}
    </List>
  );
}
