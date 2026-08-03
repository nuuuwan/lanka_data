import { Typography } from "@mui/material";

import ProgressList from "./ProgressList.js";

export default function LoadingProgress({ steps }) {
  return (
    <>
      <Typography component="h2" variant="h5">
        Loading visual
      </Typography>
      <ProgressList steps={steps} />
    </>
  );
}
