import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Typography,
} from "@mui/material";
import { QRCodeSVG } from "qrcode.react";

import { SHARE_QR_CODE_SIZE } from "../../nonview/constants/APP.js";
import styles from "./ShareQRDialog.module.css";

export default function ShareQRDialog({ open, url, onClose }) {
  return (
    <Dialog open={open} onClose={onClose} aria-labelledby="share-qr-title">
      <DialogTitle id="share-qr-title">Share with QR</DialogTitle>
      <DialogContent className={styles.content}>
        <Typography component="p" variant="body2">
          Scan to open this visual.
        </Typography>
        <QRCodeSVG
          value={url}
          size={SHARE_QR_CODE_SIZE}
          level="M"
          title="Scan to open this visual"
          className={styles.code}
        />
        <Typography component="p" variant="caption" className={styles.url}>
          {url}
        </Typography>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}
