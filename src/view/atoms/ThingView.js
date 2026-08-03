import { Chip } from "@mui/material";

export default function ThingView({ thing }) {
  const MUIIcon = thing.constructor.getMUIICON();

  return (
    <Chip
      icon={<MUIIcon />}
      label={thing.getHumanReadableValue()}
      color="info.light"
      variant="outlined"
      sx={{ m: 0.5 }}
    />
  );
}
