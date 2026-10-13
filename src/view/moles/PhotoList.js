import { ImageList, Typography } from "@mui/material";

import {
  EXAMPLE_IMAGE_LIST_COLS,
  EXAMPLE_IMAGE_LIST_GAP_PX,
  EXAMPLE_QUERIES,
} from "../../nonview/constants/ExampleQueries.js";
import ExampleQueryCard from "./ExampleQueryCard.js";
import styles from "./PhotoList.module.css";

export default function PhotoList() {
  return (
    <section aria-labelledby="examples-heading">
      <Typography component="h2" id="examples-heading" variant="h6">
        Examples
      </Typography>
      <ImageList
        className={styles.list}
        cols={EXAMPLE_IMAGE_LIST_COLS}
        gap={EXAMPLE_IMAGE_LIST_GAP_PX}
        sx={{
          gridTemplateColumns: {
            xs: "repeat(1, 1fr) !important",
            sm: "repeat(2, 1fr) !important",
            md: "repeat(3, 1fr) !important",
          },
          overflow: "visible",
        }}
      >
        {EXAMPLE_QUERIES.map((query) => (
          <ExampleQueryCard key={query} query={query} />
        ))}
      </ImageList>
    </section>
  );
}
