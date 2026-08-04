import { Chip } from "@mui/material";

export default function EntityClassView({ entityClass }) {
  const MUIIcon = entityClass.getMUIICON();

  return (
    <Chip
      icon={<MUIIcon />}
      label={entityClass.getClassName()}
      color="primary.light"
      variant="filled"
      sx={{ m: 0.5 }}
    />
  );
}
