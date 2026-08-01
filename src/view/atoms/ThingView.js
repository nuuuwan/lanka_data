import { Chip } from "@mui/material";

export default function ThingView({ thing }) {
  return (
    <Chip
      label={thing.constructor.name + "=" + thing.value}
      color="primary.light"
      variant="outlined"
      sx={{ m: 0.5 }}
    />
  );
}
