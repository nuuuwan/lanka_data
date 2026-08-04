import FunctionsIcon from "@mui/icons-material/Functions";
import { Chip } from "@mui/material";

export default function AggregateView({ aggregate }) {
  return (
    <Chip
      icon={<FunctionsIcon />}
      label={aggregate}
      variant="filled"
      sx={{ m: 0.5, backgroundColor: "info.light" }}
    />
  );
}
