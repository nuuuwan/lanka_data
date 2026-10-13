import { CardActionArea, ImageListItem, ImageListItemBar } from "@mui/material";
import { Link as RouterLink } from "react-router-dom";

import { EXAMPLE_IMAGES_PATH } from "../../nonview/constants/APP.js";
import getExampleQueryDetails from "../../nonview/core/getExampleQueryDetails.js";
import styles from "./ExampleQueryCard.module.css";

export default function ExampleQueryCard({ query }) {
  const { label, image } = getExampleQueryDetails(query);
  return (
    <ImageListItem className={styles.card}>
      <CardActionArea
        component={RouterLink}
        to={`/${query}`}
        aria-label={label}
        className={styles.action}
      >
        <img
          src={`${EXAMPLE_IMAGES_PATH}/${image}`}
          alt=""
          loading="lazy"
          className={styles.image}
        />
        <ImageListItemBar position="below" title={label} />
      </CardActionArea>
    </ImageListItem>
  );
}
