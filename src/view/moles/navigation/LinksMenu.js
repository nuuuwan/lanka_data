import BugReportOutlinedIcon from "@mui/icons-material/BugReportOutlined";
import GitHubIcon from "@mui/icons-material/GitHub";
import PersonOutlinedIcon from "@mui/icons-material/PersonOutlined";
import { ListItemIcon, Menu, MenuItem } from "@mui/material";

import {
  GITHUB_PROFILE_URL,
  GITHUB_REPOSITORY_ISSUES_URL,
  GITHUB_REPOSITORY_URL,
} from "../../../nonview/constants/APP.js";

const LINKS = [
  { label: "Repository", url: GITHUB_REPOSITORY_URL, Icon: GitHubIcon },
  {
    label: "Report a bug",
    url: GITHUB_REPOSITORY_ISSUES_URL,
    Icon: BugReportOutlinedIcon,
  },
  {
    label: "GitHub profile",
    url: GITHUB_PROFILE_URL,
    Icon: PersonOutlinedIcon,
  },
];

export default function LinksMenu({ anchor, onClose }) {
  return (
    <Menu
      id="links-menu"
      anchorEl={anchor}
      open={Boolean(anchor)}
      onClose={onClose}
    >
      {LINKS.map(({ label, url, Icon }) => (
        <MenuItem
          component="a"
          href={url}
          key={label}
          target="_blank"
          rel="noopener noreferrer"
          onClick={onClose}
        >
          <ListItemIcon>
            <Icon fontSize="small" />
          </ListItemIcon>
          {label}
        </MenuItem>
      ))}
    </Menu>
  );
}
