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

function QueryView({ query }) {
  return (
    <Box>
      <EntityClassView entityClass={query.entityClass} />
      {query.dimThingList.map((dimThing, index) => (
        <ThingView key={index} thing={dimThing} />
      ))}
      <AggregateView aggregate={query.aggregate} />
    </Box>
  );
}

export default function DatumView({ datum }) {
  return (
    <ListItem>
      <QueryView query={datum.query} />
      <SQLText> ➜ </SQLText>
      <ThingView thing={datum.answerThing} />
    </ListItem>
  );
}
