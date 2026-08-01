import { Chip } from "@mui/material";

export default function ThingView({ thing }) {
  return (
    <Chip
      label={thing.getHumanReadableValue()}
      color="info.light"
      variant="outlined"
      sx={{ m: 0.5 }}
    />
  );
}
