import DeleteOutlinedIcon from "@mui/icons-material/DeleteOutlined";
import {
  Divider,
  ListItemIcon,
  ListSubheader,
  Menu,
  MenuItem,
} from "@mui/material";

import formatRelativeTime from "../../nonview/base/formatRelativeTime.js";
import { EXAMPLE_QUERIES } from "../../nonview/constants/ExampleQueries.js";
import styles from "./QueryHistoryMenu.module.css";

function formatTimestamp(timestamp) {
  return timestamp === null
    ? "Saved previously"
    : `${new Date(timestamp).toLocaleString()} (${formatRelativeTime(timestamp)})`;
}

export default function QueryHistoryMenu({
  anchor,
  onClose,
  onOpenQuery,
  onClear,
  recentQueries,
}) {
  return (
    <Menu
      id="query-menu"
      anchorEl={anchor}
      open={Boolean(anchor)}
      onClose={onClose}
      slotProps={{ paper: { className: styles.menu } }}
    >
      <ListSubheader>Recent queries</ListSubheader>
      {recentQueries.length ? (
        recentQueries.map(({ query, timestamp }) => (
          <MenuItem key={query} onClick={() => onOpenQuery(query)}>
            <span className={styles.queryDetails}>
              <span className={styles.query}>{query}</span>
              <time className={styles.timestamp}>
                {formatTimestamp(timestamp)}
              </time>
            </span>
          </MenuItem>
        ))
      ) : (
        <MenuItem disabled>No recent queries</MenuItem>
      )}
      {recentQueries.length > 0 && (
        <MenuItem onClick={onClear}>
          <ListItemIcon>
            <DeleteOutlinedIcon fontSize="small" />
          </ListItemIcon>
          Clear recent queries
        </MenuItem>
      )}
      <Divider />
      <ListSubheader>Example queries</ListSubheader>
      {EXAMPLE_QUERIES.map(({ query }) => (
        <MenuItem key={query} onClick={() => onOpenQuery(query)}>
          <span className={styles.queryDetails}>
            <span className={styles.query}>{query}</span>
          </span>
        </MenuItem>
      ))}
    </Menu>
  );
}
