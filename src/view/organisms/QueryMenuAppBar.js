import MoreVertIcon from "@mui/icons-material/MoreVert";
import QueryStatsIcon from "@mui/icons-material/QueryStats";
import { AppBar, IconButton, Toolbar, Typography } from "@mui/material";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import RecentVisualQueries from "../../nonview/base/RecentVisualQueries.js";
import LinksMenu from "../moles/navigation/LinksMenu.js";
import QueryMenu from "../moles/navigation/QueryMenu.js";

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
        <QueryMenu
          anchor={queryMenuAnchor}
          onClear={() => setRecentQueries(RecentVisualQueries.clear())}
          onClose={() => setQueryMenuAnchor(null)}
          onOpenQuery={openQuery}
          recentQueries={recentQueries}
        />
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
        <LinksMenu
          anchor={linksMenuAnchor}
          onClose={() => setLinksMenuAnchor(null)}
        />
      </Toolbar>
    </AppBar>
  );
}
