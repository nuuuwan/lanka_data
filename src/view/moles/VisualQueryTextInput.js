import { Box, Typography } from "@mui/material";
import Prism from "prismjs";
import Editor from "react-simple-code-editor";

import { APP_FONT_FAMILY } from "../../AppTheme.js";
import {
  VISUAL_QUERY_COMPACT_INPUT_PADDING_PX,
  VISUAL_QUERY_INPUT_PADDING_PX,
} from "../../nonview/constants/APP.js";
import styles from "./VisualQueryTextInput.module.css";

const QUERY_LANGUAGE = {
  entity: {
    pattern: /^[^/]+/,
  },
  visual: {
    pattern: /(\/)[^/]+$/,
    lookbehind: true,
  },
  aggregate: {
    pattern: /(\/)[^/]+(?=\/[^/]+$)/,
    lookbehind: true,
  },
  dimension: {
    pattern: /(^|[+/])[^=+/<:,]+(?=[=:</])/,
    lookbehind: true,
  },
  value: {
    pattern: /([=:<])[^+/]+/,
    lookbehind: true,
  },
  operator: /[/+<>=,:]/,
};

function highlightVisualQuery(query) {
  return Prism.highlight(query, QUERY_LANGUAGE, "visual-query");
}

export default function VisualQueryTextInput({
  compact = false,
  disabled,
  onChange,
  onKeyDown,
  value,
}) {
  return (
    <Box>
      <Typography
        component="label"
        htmlFor="visual-query"
        variant="caption"
        className={compact ? styles.hiddenLabel : undefined}
        sx={{ color: "text.secondary", display: "block", mb: 0.5 }}
      >
        Visual query
      </Typography>
      <Box
        className={compact ? styles.compactContainer : styles.container}
        sx={{ opacity: disabled ? 0.7 : 1 }}
      >
        <Editor
          disabled={disabled}
          highlight={highlightVisualQuery}
          onKeyDown={onKeyDown}
          onValueChange={onChange}
          padding={
            compact
              ? VISUAL_QUERY_COMPACT_INPUT_PADDING_PX
              : VISUAL_QUERY_INPUT_PADDING_PX
          }
          preClassName={styles.highlight}
          style={{ fontFamily: APP_FONT_FAMILY, fontSize: "0.875rem" }}
          textareaClassName={styles.textarea}
          textareaId="visual-query"
          value={value}
        />
      </Box>
      {!compact && (
        <Typography
          variant="caption"
          sx={{ color: "text.secondary", display: "block", mt: 0.5 }}
        >
          Press Enter to update
        </Typography>
      )}
    </Box>
  );
}
