import { Alert, AlertTitle, Stack } from "@mui/material";

export default function VisualQueryEditorHelp() {
  return (
    <Stack spacing={1} sx={{ mb: 2 }}>
      <Alert severity="info">
        <AlertTitle>Choose your data and visual</AlertTitle>
        Select a data type under What data?, then choose a chart, map, or table
        under Show as. Count is currently the only calculation.
      </Alert>
      <Alert severity="info">
        <AlertTitle>Group or filter with facets</AlertTitle>
        Choose a Field and leave its Operator as None to group results by that
        field. Use = and choose a Value to filter results. Use AND to add
        another facet, or the remove button to delete one.
      </Alert>
      <Alert severity="info">
        <AlertTitle>Apply your changes</AlertTitle>
        Select Update visualization, or press Enter in a text field, to apply
        your choices and close the drawer. Close hides the drawer without
        applying changes; your edits remain in the query bar.
      </Alert>
    </Stack>
  );
}
