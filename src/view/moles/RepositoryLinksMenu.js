import BugReportOutlinedIcon from "@mui/icons-material/BugReportOutlined";
import GitHubIcon from "@mui/icons-material/GitHub";
import PersonOutlinedIcon from "@mui/icons-material/PersonOutlined";
import { ListItemIcon, Menu, MenuItem } from "@mui/material";

import {
  GITHUB_PROFILE_URL,
  GITHUB_REPOSITORY_ISSUES_URL,
  GITHUB_REPOSITORY_URL,
} from "../../nonview/constants/APP.js";

const LINKS = [
  { href: GITHUB_REPOSITORY_URL, icon: GitHubIcon, label: "Repository" },
  {
    href: GITHUB_REPOSITORY_ISSUES_URL,
    icon: BugReportOutlinedIcon,
    label: "Report a bug",
  },
  {
    href: GITHUB_PROFILE_URL,
    icon: PersonOutlinedIcon,
    label: "GitHub profile",
  },
];

export default function RepositoryLinksMenu({ anchor, onClose }) {
  return (
    <Menu
      id="links-menu"
      anchorEl={anchor}
      open={Boolean(anchor)}
      onClose={onClose}
    >
      {LINKS.map(({ href, icon: Icon, label }) => (
        <MenuItem
          component="a"
          href={href}
          key={href}
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
