import FormatUtils from "../moles/visual_utils/FormatUtils.js";
import styles from "./ProgressList.module.css";

function getStatusText(step) {
  if (step.status === "complete" || step.status === "active") {
    return FormatUtils.humanizeDuration(step.durationSeconds);
  }
  return "Waiting";
}

function StatusIcon({ status }) {
  return (
    <span
      className={`${styles.status} ${styles[status]}`}
      aria-label={status}
      role="img"
    />
  );
}

export default function ProgressList({ steps }) {
  return (
    <ul className={styles.root} aria-label="Visual loading progress">
      {steps.map((step) => (
        <li className={styles.step} key={step.label}>
          <StatusIcon status={step.status} />
          <span className={styles.label}>{step.label}</span>
          <span className={styles.duration}>{getStatusText(step)}</span>
        </li>
      ))}
    </ul>
  );
}
