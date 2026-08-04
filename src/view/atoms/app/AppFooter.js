import { Box, Link, Typography } from "@mui/material";
import GitHubIcon from "@mui/icons-material/GitHub";
import { QRCodeSVG } from "qrcode.react";

import { AppTheme } from "../../../AppTheme.js";
import VERSION from "../../../nonview/cons/VERSION.js";
import { APP_QR_CODE_SIZE, APP_URL } from "../../../nonview/constants/APP.js";
import styles from "./AppFooter.module.css";

export default function AppFooter() {
  return (
    <Box sx={{ m: 2, textAlign: "center" }}>
      <Typography
        variant="caption"
        sx={{
          color: "info.main",
          display: "inline-flex",
          alignItems: "center",
          gap: 0.5,
        }}
      >
        v{VERSION.DATETIME_STR} by{" "}
        <Link
          href="https://github.com/nuuuwan/lanka_data"
          target="_blank"
          rel="noopener noreferrer"
          sx={{
            color: "info.main",
            display: "inline-flex",
            alignItems: "center",
            gap: 0.25,
          }}
        >
          <GitHubIcon sx={{ fontSize: "0.875rem" }} />
          @nuuuwan
        </Link>
      </Typography>
      <br />
      <Box sx={{ m: 2 }}>
        <Link
          aria-label="Open Lanka Data"
          className={styles.qrCodeLink}
          href={APP_URL}
        >
          <QRCodeSVG
            bgColor="#ffffff"
            fgColor={AppTheme.palette.info.main}
            level="M"
            size={APP_QR_CODE_SIZE}
            title="Scan to open Lanka Data"
            value={APP_URL}
          />
        </Link>
      </Box>
    </Box>
  );
}
