import { useParams } from "react-router-dom";
import { Typography } from "@mui/material";

export default function QueryPage() {
  const { "*": queryStr } = useParams();
  return (
    <Typography variant="body1" sx={{ mt: 2 }}>
      {queryStr}
    </Typography>
  );
}
