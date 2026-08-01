import { ListItem } from "@mui/material";

import AggregateView from "../atoms/AggregateView.js";
import EntityClassView from "../atoms/EntityClassView.js";
import ThingView from "../atoms/ThingView.js";

export default function DatumView({ datum }) {
  return (
    <ListItem>
      <AggregateView aggregate={datum.aggregate} />
      {" of "}
      <EntityClassView entityClass={datum.entityClass} />
      {" where "}
      {datum.dimThingList.map((dimThing, index) => (
        <ThingView key={index} thing={dimThing} />
      ))}
      {" = "}
      <ThingView thing={datum.cellThing} />
    </ListItem>
  );
}
