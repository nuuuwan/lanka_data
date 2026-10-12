import { useState } from "react";

import { copyTextToClipboard } from "../base/Clipboard.js";

export default function useVisualQueryActions(
  queryOptions,
  onChange,
  onSubmit,
) {
  const [shareFeedback, setShareFeedback] = useState(null);
  const metadataKeyLists = queryOptions.metadataKeyLists.filter(
    (keys) => keys.length > 0,
  );

  async function copyShareLink() {
    try {
      await copyTextToClipboard(window.location.href);
      setShareFeedback("Share link copied");
    } catch {
      setShareFeedback("Could not copy share link");
    }
  }

  function selectRandomQuery() {
    const keys =
      metadataKeyLists[Math.floor(Math.random() * metadataKeyLists.length)];
    const query = `${keys[Math.floor(Math.random() * keys.length)]}/Table`;
    onChange(query);
    onSubmit(query);
  }

  return {
    shareFeedback,
    dismissShareFeedback: () => setShareFeedback(null),
    copyShareLink,
    selectRandomQuery,
    randomDisabled: metadataKeyLists.length === 0,
  };
}
