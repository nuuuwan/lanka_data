import { Box } from "@mui/material";

import AggregateView from "../../../atoms/AggregateView.js";
import EntityClassView from "../../../atoms/EntityClassView.js";
import ThingView from "../../../atoms/ThingView.js";

export default function QueryView({ query }) {
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
