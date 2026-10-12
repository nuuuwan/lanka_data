import MoreVertIcon from "@mui/icons-material/MoreVert";
import QueryStatsIcon from "@mui/icons-material/QueryStats";
import { AppBar, IconButton, Toolbar, Typography } from "@mui/material";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import RecentVisualQueries from "../../nonview/base/RecentVisualQueries.js";
import AppLinksMenu from "../moles/AppLinksMenu.js";
import RecentQueriesMenu from "../moles/RecentQueriesMenu.js";
import ShareQRDialog from "../moles/ShareQRDialog.js";

export default function QueryMenuAppBar({ loadedVisualQuery }) {
  const navigate = useNavigate();
  const [queryMenuAnchor, setQueryMenuAnchor] = useState(null);
  const [linksMenuAnchor, setLinksMenuAnchor] = useState(null);
  const [shareURL, setShareURL] = useState(null);
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
      <Toolbar
        sx={{
          height: "var(--app-bar-height)",
          minHeight: "var(--app-bar-height) !important",
        }}
      >
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
        <RecentQueriesMenu
          anchorEl={queryMenuAnchor}
          onClose={() => setQueryMenuAnchor(null)}
          recentQueries={recentQueries}
          onOpenQuery={openQuery}
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
        <AppLinksMenu
          anchorEl={linksMenuAnchor}
          onClose={() => setLinksMenuAnchor(null)}
          onShareQR={() => {
            setLinksMenuAnchor(null);
            setShareURL(window.location.href);
          }}
        />
        <ShareQRDialog
          open={shareURL !== null}
          url={shareURL || ""}
          onClose={() => setShareURL(null)}
        />
      </Toolbar>
    </AppBar>
  );
}
