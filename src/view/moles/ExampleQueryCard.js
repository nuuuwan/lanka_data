import {
  Card,
  CardActionArea,
  CardContent,
  CardMedia,
  Typography,
} from "@mui/material";
import { Link as RouterLink } from "react-router-dom";

import { EXAMPLE_IMAGES_PATH } from "../../nonview/constants/APP.js";
import styles from "./ExampleQueryCard.module.css";

export default function ExampleQueryCard({ description, image, query }) {
  return (
    <Card component="li" elevation={0} className={styles.card}>
      <CardActionArea
        component={RouterLink}
        to={`/${query}`}
        className={styles.action}
      >
        <CardMedia
          component="img"
          image={`${EXAMPLE_IMAGES_PATH}/${image}`}
          alt=""
          loading="lazy"
          className={styles.image}
        />
        <CardContent className={styles.content}>
          <Typography component="h3" variant="body1">
            {description}
          </Typography>
        </CardContent>
      </CardActionArea>
    </Card>
  );
}
