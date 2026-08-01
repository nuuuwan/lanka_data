import { ListItem } from "@mui/material";

import SQLText from "../atoms/SQLText.js";
import QueryView from "./QueryView.js";
import ThingView from "../atoms/ThingView.js";

export default function DatumView({ datum }) {
  return (
    <ListItem>
      <QueryView query={datum.query} />
      <SQLText> ➜ </SQLText>
      <ThingView thing={datum.answerThing} />
    </ListItem>
  );
}
