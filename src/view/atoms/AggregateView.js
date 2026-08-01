import { Chip } from "@mui/material";

export default function AggregateView({ aggregate }) {
  return (
    <Chip
      label={aggregate}
      color="secondary.main"
      variant="filled"
      sx={{ m: 0.5 }}
    />
  );
}
