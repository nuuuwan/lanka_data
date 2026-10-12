import { Typography } from "@mui/material";

import { EXAMPLE_QUERIES } from "../../nonview/constants/ExampleQueries.js";
import ExampleQueryCard from "./ExampleQueryCard.js";
import styles from "./ExampleQueryLinks.module.css";

export default function ExampleQueryLinks() {
  return (
    <section aria-labelledby="examples-heading">
      <Typography component="h2" id="examples-heading" variant="h6">
        Examples
      </Typography>
      <ul className={styles.list}>
        {EXAMPLE_QUERIES.map((query) => (
          <ExampleQueryCard key={query} query={query} />
        ))}
      </ul>
    </section>
  );
}
