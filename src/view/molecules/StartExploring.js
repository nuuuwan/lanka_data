import { Link as MuiLink, Typography } from "@mui/material";
import { Link as RouterLink } from "react-router-dom";

import { START_EXPLORING_STEPS } from "../../nonview/constants/StartExploring.js";
import styles from "./StartExploring.module.css";

export default function StartExploring() {
  return (
    <section
      aria-labelledby="start-exploring-heading"
      className={styles.section}
    >
      <Typography component="h2" id="start-exploring-heading" variant="h6">
        Start exploring
      </Typography>
      <ol className={styles.steps}>
        {START_EXPLORING_STEPS.map(
          ({ question, interpretation, query }) => (
            <li className={styles.step} key={query}>
              <MuiLink component={RouterLink} to={`/${query}`}>
                {question}
              </MuiLink>
              <Typography variant="body2">{interpretation}</Typography>
            </li>
          ),
        )}
      </ol>
    </section>
  );
}
