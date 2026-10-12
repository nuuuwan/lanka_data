import BugReportOutlinedIcon from "@mui/icons-material/BugReportOutlined";
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
} from "../../nonview/constants/APP.js";

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
      <MenuItem
        component="a"
        href={GITHUB_REPOSITORY_URL}
        target="_blank"
        rel="noopener noreferrer"
        onClick={onClose}
      >
        <ListItemIcon>
          <GitHubIcon fontSize="small" />
        </ListItemIcon>
        Repository
      </MenuItem>
      <MenuItem
        component="a"
        href={GITHUB_REPOSITORY_ISSUES_URL}
        target="_blank"
        rel="noopener noreferrer"
        onClick={onClose}
      >
        <ListItemIcon>
          <BugReportOutlinedIcon fontSize="small" />
        </ListItemIcon>
        Report a bug
      </MenuItem>
      <MenuItem
        component="a"
        href={GITHUB_PROFILE_URL}
        target="_blank"
        rel="noopener noreferrer"
        onClick={onClose}
      >
        <ListItemIcon>
          <PersonOutlinedIcon fontSize="small" />
        </ListItemIcon>
        GitHub profile
      </MenuItem>
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
