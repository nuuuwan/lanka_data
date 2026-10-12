import { Link as MuiLink, Typography } from "@mui/material";
import { Link as RouterLink } from "react-router-dom";

import { EXAMPLE_QUERIES } from "../../nonview/constants/ExampleQueries.js";
import styles from "./ExampleQueryLinks.module.css";

export default function ExampleQueryLinks() {
  return (
    <section aria-labelledby="examples-heading">
      <Typography component="h2" id="examples-heading" variant="h6">
        Examples
      </Typography>
      <ul className={styles.list}>
        {EXAMPLE_QUERIES.map(({ label, description, query }) => (
          <li className={styles.example} key={query}>
            <MuiLink component={RouterLink} to={`/${query}`}>
              {label}
            </MuiLink>
            <Typography variant="body2">{description}</Typography>
          </li>
        ))}
      </ul>
    </section>
  );
}
