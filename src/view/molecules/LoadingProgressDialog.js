import { Dialog, DialogContent, DialogTitle } from "@mui/material";

import ProgressList from "./ProgressList.js";

export default function LoadingProgressDialog({ steps }) {
  return (
    <Dialog open aria-labelledby="loading-progress-dialog-title">
      <DialogTitle id="loading-progress-dialog-title">
        Loading visual
      </DialogTitle>
      <DialogContent>
        <ProgressList steps={steps} />
      </DialogContent>
    </Dialog>
  );
}
