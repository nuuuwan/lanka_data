import { ListItem, Box } from "@mui/material";

import AggregateView from "../atoms/AggregateView.js";
import EntityClassView from "../atoms/EntityClassView.js";
import ThingView from "../atoms/ThingView.js";

function SQLText({ children }) {
  return (
    <Box component="span" sx={{ color: "primary.light" }}>
      {children}
    </Box>
  );
}

export default function DatumView({ datum }) {
  return (
    <ListItem>
      <EntityClassView entityClass={datum.entityClass} />
      {datum.dimThingList.map((dimThing, index) => (
        <ThingView key={index} thing={dimThing} />
      ))}
      <AggregateView aggregate={datum.aggregate} />
      <SQLText> ➜ </SQLText>
      <ThingView thing={datum.cellThing} />
    </ListItem>
  );
}
