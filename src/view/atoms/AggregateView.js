import { Chip } from "@mui/material";

export default function AggregateView({ aggregate }) {
  return (
    <Chip
      label={aggregate}
      variant="filled"
      sx={{ m: 0.5, color: "white", backgroundColor: "info.light" }}
    />
  );
}
