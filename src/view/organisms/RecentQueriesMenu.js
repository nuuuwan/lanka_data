import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import HistoryIcon from "@mui/icons-material/History";
import {
  Divider,
  IconButton,
  ListItemIcon,
  Menu,
  MenuItem,
  Tooltip,
} from "@mui/material";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import RecentVisualQueries from "../../nonview/base/RecentVisualQueries.js";
import styles from "./RecentQueriesMenu.module.css";

export default function RecentQueriesMenu({ loadedVisualQuery }) {
  const navigate = useNavigate();
  const [anchorElement, setAnchorElement] = useState(null);
  const [queries, setQueries] = useState(() => RecentVisualQueries.read());

  useEffect(() => {
    if (loadedVisualQuery) {
      setQueries(RecentVisualQueries.add(loadedVisualQuery));
    }
  }, [loadedVisualQuery]);

  function openQuery(query) {
    setAnchorElement(null);
    navigate(`/${query}`);
  }

  function clearQueries() {
    setQueries(RecentVisualQueries.clear());
    setAnchorElement(null);
  }

  return (
    <div className={styles.root}>
      <Tooltip title="Recent queries">
        <IconButton
          aria-label="Recent queries"
          size="small"
          onClick={(event) => setAnchorElement(event.currentTarget)}
        >
          <HistoryIcon fontSize="small" />
        </IconButton>
      </Tooltip>
      <Menu
        anchorEl={anchorElement}
        open={Boolean(anchorElement)}
        onClose={() => setAnchorElement(null)}
      >
        {queries.length === 0 ? (
          <MenuItem disabled>No recent queries</MenuItem>
        ) : (
          queries.map((query) => (
            <MenuItem key={query} onClick={() => openQuery(query)}>
              <span className={styles.query}>{query}</span>
            </MenuItem>
          ))
        )}
        {queries.length > 0 && <Divider />}
        {queries.length > 0 && (
          <MenuItem onClick={clearQueries}>
            <ListItemIcon>
              <DeleteOutlineIcon fontSize="small" />
            </ListItemIcon>
            Clear recent queries
          </MenuItem>
        )}
      </Menu>
    </div>
  );
}
