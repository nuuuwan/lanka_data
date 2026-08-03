import FaceIcon from "@mui/icons-material/Face";
import { Chip } from "@mui/material";

export default function EntityClassView({ entityClass }) {
  return (
    <Chip
      icon={<FaceIcon />}
      label={entityClass.getClassName()}
      color="primary.light"
      variant="filled"
      sx={{ m: 0.5 }}
    />
  );
}
