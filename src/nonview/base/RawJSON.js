import { JSON_DOWNLOAD_FILE_NAME } from "../constants/APP.js";

export function getRawJSONURL(location) {
  const pathname = location.pathname.replace(/\/$/, "");
  return `${location.origin}${pathname}/${JSON_DOWNLOAD_FILE_NAME}`;
}
