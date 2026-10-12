import { ListSubheader, MenuItem, TextField } from "@mui/material";

import { VISUAL_GROUPS } from "../../nonview/constants/VisualQueryOptions.js";
import VisualFactory from "./visuals/VisualFactory.js";
import { getVisualLabel } from "./LaypersonQueryUtils.js";

export default function LaypersonQueryFields({
  parts,
  entityOptions,
  onUpdate,
  step,
}) {
  return (
    <>
      {step === "Data" && (
        <TextField
          select
          label="What data?"
          size="small"
          value={parts.entity}
          onChange={(event) => onUpdate("entity", event.target.value)}
        >
          {entityOptions.map((entity) => (
            <MenuItem key={entity} value={entity}>
              {entity}
            </MenuItem>
          ))}
        </TextField>
      )}
      {step === "Visual" && (
        <TextField
          select
          label="Show as"
          size="small"
          value={parts.visual}
          onChange={(event) => onUpdate("visual", event.target.value)}
        >
          {VISUAL_GROUPS.flatMap((group) => [
            <ListSubheader key={group.label}>{group.label}</ListSubheader>,
            ...group.visuals
              .filter((visual) => VisualFactory.list().includes(visual))
              .map((visual) => (
                <MenuItem key={visual} sx={{ pl: 3 }} value={visual}>
                  {getVisualLabel(visual)}
                </MenuItem>
              )),
          ])}
        </TextField>
      )}
    </>
  );
}
