import { ListSubheader, MenuItem, TextField } from "@mui/material";

import { VISUAL_GROUPS } from "../../nonview/constants/VisualQueryOptions.js";
import VisualFactory from "./visuals/VisualFactory.js";
import { getVisualLabel } from "./LaypersonQueryUtils.js";

export default function LaypersonQueryFields({
  parts,
  entityOptions,
  onUpdate,
  onKeyDown,
}) {
  return (
    <>
      <TextField
        select
        label="What data?"
        size="small"
        value={parts.entity}
        onChange={(event) => onUpdate("entity", event.target.value)}
        helperText="Choose the type of data"
      >
        {entityOptions.map((entity) => (
          <MenuItem key={entity} value={entity}>
            {entity}
          </MenuItem>
        ))}
      </TextField>
      <TextField
        label="Calculate"
        size="small"
        value={parts.aggregate}
        onChange={(event) => onUpdate("aggregate", event.target.value)}
        onKeyDown={onKeyDown}
        helperText="For example, Count"
      />
      <TextField
        select
        label="Show as"
        size="small"
        value={parts.visual}
        onChange={(event) => onUpdate("visual", event.target.value)}
        helperText="Choose a visual"
      >
        {VISUAL_GROUPS.flatMap((group) => [
          <ListSubheader key={group.label}>{group.label}</ListSubheader>,
          ...group.visuals
            .filter((visual) => VisualFactory.list().includes(visual))
            .map((visual) => (
              <MenuItem key={visual} value={visual}>
                {getVisualLabel(visual)}
              </MenuItem>
            )),
        ])}
      </TextField>
    </>
  );
}
