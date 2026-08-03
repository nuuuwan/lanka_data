import { ListSubheader, MenuItem } from "@mui/material";

import { FIELD_GROUPS } from "../../nonview/constants/VisualQueryOptions.js";
import { getVisualLabel } from "../moles/LaypersonQueryUtils.js";

export default function FieldOptions({ fields }) {
  const groupedFields = FIELD_GROUPS.map((group) => ({
    label: group.label,
    fields: fields.filter((field) => group.fields.includes(field)),
  })).filter((group) => group.fields.length);
  const knownFields = new Set(FIELD_GROUPS.flatMap((group) => group.fields));
  const otherFields = fields.filter((field) => !knownFields.has(field));

  if (otherFields.length) {
    groupedFields.push({ label: "Other", fields: otherFields });
  }

  return groupedFields.flatMap((group) => [
    <ListSubheader key={group.label}>{group.label}</ListSubheader>,
    ...group.fields.map((field) => (
      <MenuItem key={field} value={field}>
        {getVisualLabel(field)}
      </MenuItem>
    )),
  ]);
}
