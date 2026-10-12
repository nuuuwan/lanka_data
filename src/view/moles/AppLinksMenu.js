import BugReportOutlinedIcon from "@mui/icons-material/BugReportOutlined";
import CodeIcon from "@mui/icons-material/Code";
import GitHubIcon from "@mui/icons-material/GitHub";
import PersonOutlinedIcon from "@mui/icons-material/PersonOutlined";
import QrCodeIcon from "@mui/icons-material/QrCode";
import RefreshIcon from "@mui/icons-material/Refresh";
import {
  Divider,
  ListItemIcon,
  Menu,
  MenuItem,
  Typography,
} from "@mui/material";

import VERSION from "../../nonview/cons/VERSION.js";
import {
  GITHUB_PROFILE_URL,
  GITHUB_REPOSITORY_ISSUES_URL,
  GITHUB_REPOSITORY_URL,
  PROGRAMMATIC_DATA_DOCS_URL,
} from "../../nonview/constants/APP.js";
import AppExternalLinkMenuItem from "./AppExternalLinkMenuItem.js";

export default function AppLinksMenu({
  anchorEl,
  onClose,
  onShareQR,
  onRefresh,
}) {
  return (
    <Menu
      id="links-menu"
      anchorEl={anchorEl}
      open={Boolean(anchorEl)}
      onClose={onClose}
    >
      <AppExternalLinkMenuItem
        href={GITHUB_REPOSITORY_URL}
        onClose={onClose}
        icon={GitHubIcon}
      >
        Repository
      </AppExternalLinkMenuItem>
      <AppExternalLinkMenuItem
        href={PROGRAMMATIC_DATA_DOCS_URL}
        onClose={onClose}
        icon={CodeIcon}
      >
        Access data Programmatically
      </AppExternalLinkMenuItem>
      <AppExternalLinkMenuItem
        href={GITHUB_REPOSITORY_ISSUES_URL}
        onClose={onClose}
        icon={BugReportOutlinedIcon}
      >
        Report a bug
      </AppExternalLinkMenuItem>
      <AppExternalLinkMenuItem
        href={GITHUB_PROFILE_URL}
        onClose={onClose}
        icon={PersonOutlinedIcon}
      >
        GitHub profile
      </AppExternalLinkMenuItem>
      <MenuItem onClick={onShareQR}>
        <ListItemIcon>
          <QrCodeIcon fontSize="small" />
        </ListItemIcon>
        Share with QR
      </MenuItem>
      <MenuItem onClick={onRefresh}>
        <ListItemIcon>
          <RefreshIcon fontSize="small" />
        </ListItemIcon>
        Refresh
      </MenuItem>
      <Divider />
      <Typography
        component="li"
        variant="caption"
        sx={{ color: "text.secondary", listStyle: "none", px: 2, py: 1 }}
      >
        v{VERSION.DATETIME_STR} by @nuuuwan
      </Typography>
    </Menu>
  );
}
