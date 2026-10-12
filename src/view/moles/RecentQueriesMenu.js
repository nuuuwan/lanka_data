import { Divider, ListSubheader, Menu, MenuItem } from "@mui/material";

import formatRelativeTime from "../../nonview/base/formatRelativeTime.js";
import { EXAMPLE_QUERIES } from "../../nonview/constants/ExampleQueries.js";
import styles from "./RecentQueriesMenu.module.css";

function formatTimestamp(timestamp) {
  return timestamp === null
    ? "Saved previously"
    : `${new Date(timestamp).toLocaleString()} (${formatRelativeTime(timestamp)})`;
}

export default function RecentQueriesMenu({
  anchorEl,
  onClose,
  recentQueries,
  onOpenQuery,
}) {
  return (
    <Menu
      id="query-menu"
      anchorEl={anchorEl}
      open={Boolean(anchorEl)}
      onClose={onClose}
      slotProps={{ paper: { className: styles.menu } }}
    >
      <ListSubheader>Recent queries</ListSubheader>
      {recentQueries.length === 0 ? (
        <MenuItem disabled>No recent queries</MenuItem>
      ) : (
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
      )}
      {recentQueries.length < 5 && (
        <>
          <Divider />
          <ListSubheader>Example queries</ListSubheader>
          {EXAMPLE_QUERIES.map(({ query }) => (
            <MenuItem key={query} onClick={() => onOpenQuery(query)}>
              <span className={styles.queryDetails}>
                <span className={styles.query}>{query}</span>
              </span>
            </MenuItem>
          ))}
        </>
      )}
    </Menu>
  );
}
