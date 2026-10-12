import { useState } from "react";

import { copyTextToClipboard } from "../base/Clipboard.js";

export default function useVisualQueryActions() {
  const [shareFeedback, setShareFeedback] = useState(null);

  async function copyShareLink() {
    try {
      await copyTextToClipboard(window.location.href);
      setShareFeedback("Share link copied");
    } catch {
      setShareFeedback("Could not copy share link");
    }
  }

  return {
    shareFeedback,
    dismissShareFeedback: () => setShareFeedback(null),
    copyShareLink,
  };
}
