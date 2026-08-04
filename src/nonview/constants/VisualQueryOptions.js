import { FIELD_GROUPS } from "./FieldGroups.js";

export { FIELD_GROUPS } from "./FieldGroups.js";
export { VISUAL_GROUPS } from "./VisualGroups.js";

export const DIMENSION_OPERATORS = [
  { value: "", label: "None" },
  { value: "=", label: "=" },
  { value: ":", label: ":" },
  { value: "<", label: "<" },
];

export function getFieldGroups(fields) {
  const knownFields = new Set(FIELD_GROUPS.flatMap((group) => group.fields));
  return [
    ...FIELD_GROUPS.map((group) => ({
      label: group.label,
      fields: fields.filter((field) => group.fields.includes(field)),
    })),
    {
      label: "Other",
      fields: fields.filter((field) => !knownFields.has(field)),
    },
  ].filter((group) => group.fields.length);
}
