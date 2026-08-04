import BugReportOutlinedIcon from "@mui/icons-material/BugReportOutlined";
import GitHubIcon from "@mui/icons-material/GitHub";
import MoreVertIcon from "@mui/icons-material/MoreVert";
import PersonOutlinedIcon from "@mui/icons-material/PersonOutlined";
import QueryStatsIcon from "@mui/icons-material/QueryStats";
import {
  AppBar,
  Divider,
  IconButton,
  ListItemIcon,
  ListSubheader,
  Menu,
  MenuItem,
  Toolbar,
  Typography,
} from "@mui/material";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import formatRelativeTime from "../../nonview/base/formatRelativeTime.js";
import RecentVisualQueries from "../../nonview/base/RecentVisualQueries.js";
import {
  GITHUB_PROFILE_URL,
  GITHUB_REPOSITORY_ISSUES_URL,
  GITHUB_REPOSITORY_URL,
} from "../../nonview/constants/APP.js";
import { EXAMPLE_QUERIES } from "../../nonview/constants/ExampleQueries.js";
import styles from "./QueryMenuAppBar.module.css";

function formatTimestamp(timestamp) {
  return timestamp === null
    ? "Saved previously"
    : `${new Date(timestamp).toLocaleString()} (${formatRelativeTime(timestamp)})`;
}

export default function QueryMenuAppBar({ loadedVisualQuery }) {
  const navigate = useNavigate();
  const [queryMenuAnchor, setQueryMenuAnchor] = useState(null);
  const [linksMenuAnchor, setLinksMenuAnchor] = useState(null);
  const [recentQueries, setRecentQueries] = useState(() =>
    RecentVisualQueries.read(),
  );

  useEffect(() => {
    if (loadedVisualQuery) {
      setRecentQueries(RecentVisualQueries.add(loadedVisualQuery));
    }
  }, [loadedVisualQuery]);

  function openQuery(query) {
    setQueryMenuAnchor(null);
    navigate(`/${query}`);
  }

  return (
    <AppBar position="sticky" sx={{ top: 0 }}>
      <Toolbar>
        <Typography component="h1" variant="h6" sx={{ flexGrow: 1 }}>
          Lanka Data
        </Typography>
        <IconButton
          aria-label="Open query menu"
          color="inherit"
          aria-controls={queryMenuAnchor ? "query-menu" : undefined}
          aria-haspopup="true"
          aria-expanded={queryMenuAnchor ? "true" : undefined}
          onClick={(event) => setQueryMenuAnchor(event.currentTarget)}
        >
          <QueryStatsIcon />
        </IconButton>
        <Menu
          id="query-menu"
          anchorEl={queryMenuAnchor}
          open={Boolean(queryMenuAnchor)}
          onClose={() => setQueryMenuAnchor(null)}
          slotProps={{ paper: { className: styles.menu } }}
        >
          <ListSubheader>Recent queries</ListSubheader>
          {recentQueries.length === 0 ? (
            <MenuItem disabled>No recent queries</MenuItem>
          ) : (
            recentQueries.map(({ query, timestamp }) => (
              <MenuItem key={query} onClick={() => openQuery(query)}>
                <span className={styles.queryDetails}>
                  <span className={styles.query}>{query}</span>
                  <time className={styles.timestamp}>
                    {formatTimestamp(timestamp)}
                  </time>
                </span>
              </MenuItem>
            ))
          )}
          {recentQueries.length < 5 && (
            <>
              <Divider />
              <ListSubheader>Example queries</ListSubheader>
              {EXAMPLE_QUERIES.map(({ query }) => (
                <MenuItem key={query} onClick={() => openQuery(query)}>
                  <span className={styles.queryDetails}>
                    <span className={styles.query}>{query}</span>
                  </span>
                </MenuItem>
              ))}
            </>
          )}
        </Menu>
        <IconButton
          aria-label="Open links menu"
          color="inherit"
          aria-controls={linksMenuAnchor ? "links-menu" : undefined}
          aria-haspopup="true"
          aria-expanded={linksMenuAnchor ? "true" : undefined}
          onClick={(event) => setLinksMenuAnchor(event.currentTarget)}
        >
          <MoreVertIcon />
        </IconButton>
        <Menu
          id="links-menu"
          anchorEl={linksMenuAnchor}
          open={Boolean(linksMenuAnchor)}
          onClose={() => setLinksMenuAnchor(null)}
        >
          <MenuItem
            component="a"
            href={GITHUB_REPOSITORY_URL}
            target="_blank"
            rel="noopener noreferrer"
            onClick={() => setLinksMenuAnchor(null)}
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
            onClick={() => setLinksMenuAnchor(null)}
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
            onClick={() => setLinksMenuAnchor(null)}
          >
            <ListItemIcon>
              <PersonOutlinedIcon fontSize="small" />
            </ListItemIcon>
            GitHub profile
          </MenuItem>
        </Menu>
      </Toolbar>
    </AppBar>
  );
}
