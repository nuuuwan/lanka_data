import DeleteOutlinedIcon from "@mui/icons-material/DeleteOutlined";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import {
  AppBar,
  Button,
  Divider,
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
import { EXAMPLE_QUERIES } from "../../nonview/constants/ExampleQueries.js";
import styles from "./QueryMenuAppBar.module.css";

function formatTimestamp(timestamp) {
  return timestamp === null
    ? "Saved previously"
    : `${new Date(timestamp).toLocaleString()} (${formatRelativeTime(timestamp)})`;
}

export default function QueryMenuAppBar({ loadedVisualQuery }) {
  const navigate = useNavigate();
  const [anchorElement, setAnchorElement] = useState(null);
  const [recentQueries, setRecentQueries] = useState(() =>
    RecentVisualQueries.read(),
  );

  useEffect(() => {
    if (loadedVisualQuery) {
      setRecentQueries(RecentVisualQueries.add(loadedVisualQuery));
    }
  }, [loadedVisualQuery]);

  function openQuery(query) {
    setAnchorElement(null);
    navigate(`/${query}`);
  }

  function clearRecentQueries() {
    setRecentQueries(RecentVisualQueries.clear());
  }

  return (
    <AppBar position="static">
      <Toolbar>
        <Typography component="h1" variant="h6" sx={{ flexGrow: 1 }}>
          Lanka Data
        </Typography>
        <Button
          color="inherit"
          endIcon={<ExpandMoreIcon />}
          aria-controls={anchorElement ? "query-menu" : undefined}
          aria-haspopup="true"
          aria-expanded={anchorElement ? "true" : undefined}
          onClick={(event) => setAnchorElement(event.currentTarget)}
        >
          Queries
        </Button>
        <Menu
          id="query-menu"
          anchorEl={anchorElement}
          open={Boolean(anchorElement)}
          onClose={() => setAnchorElement(null)}
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
          {recentQueries.length > 0 && (
            <MenuItem onClick={clearRecentQueries}>
              <ListItemIcon>
                <DeleteOutlinedIcon fontSize="small" />
              </ListItemIcon>
              Clear recent queries
            </MenuItem>
          )}
          <Divider />
          <ListSubheader>Example queries</ListSubheader>
          {EXAMPLE_QUERIES.map(({ query }) => (
            <MenuItem key={query} onClick={() => openQuery(query)}>
              <span className={styles.queryDetails}>
                <span className={styles.query}>{query}</span>
              </span>
            </MenuItem>
          ))}
        </Menu>
      </Toolbar>
    </AppBar>
  );
}
