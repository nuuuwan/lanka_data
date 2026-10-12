import { Typography } from "@mui/material";

import ExampleQueryLinks from "../moles/ExampleQueryLinks.js";
import QueryMenuAppBar from "../organisms/QueryMenuAppBar.js";
import styles from "./HomePage.module.css";

export default function HomePage() {
  return (
    <>
      <QueryMenuAppBar />
      <main className={styles.page}>
        <Typography component="h2" variant="h5">
          Explore Sri Lanka through data
        </Typography>
        <Typography className={styles.introduction}>
          One query, one tool for people and code. Start with an example to
          explore census and election data through charts, maps, and tables.
        </Typography>
        <ExampleQueryLinks />
      </main>
    </>
  );
}
