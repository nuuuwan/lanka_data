import FilterAltIcon from "@mui/icons-material/FilterAlt";
import { Chip } from "@mui/material";

export default function ThingView({ thing }) {
  return (
    <Chip
      icon={<FilterAltIcon />}
      label={thing.getHumanReadableValue()}
      color="info.light"
      variant="outlined"
      sx={{ m: 0.5 }}
    />
  );
}
