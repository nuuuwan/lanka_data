import FaceIcon from "@mui/icons-material/Face";
import { Chip } from "@mui/material";

export default function EntityClassView({ entityClass }) {
  return (
    <Chip
      icon={<FaceIcon />}
      label={entityClass.name}
      color="primary.main"
      variant="filled"
      sx={{ m: 0.5 }}
    />
  );
}
