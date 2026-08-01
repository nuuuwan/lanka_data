import React from "react";
import FaceIcon from "@mui/icons-material/Face";
import { Chip, ListItem } from "@mui/material";

function EntityClassView({ entityClass }) {
  return (
    <Chip
      icon={<FaceIcon />}
      label={entityClass.name}
      color="primary"
      variant="filled"
      sx={{ m: 0.5 }}
    />
  );
}

function ThingView({ thing }) {
  return (
    <Chip
      label={thing.constructor.name + "=" + thing.value}
      color="secondary"
      variant="outlined"
      sx={{ m: 0.5 }}
    />
  );
}

function AggregateView({ aggregate }) {
  return (
    <Chip label={aggregate} color="success" variant="filled" sx={{ m: 0.5 }} />
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
      <ThingView thing={datum.cellThing} />
    </ListItem>
  );
}
