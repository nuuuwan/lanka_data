import {
  JSON_DATA_URL_PREFIX,
  JSON_DOWNLOAD_FILE_NAME,
} from "../constants/APP.js";

export function getJSONDownloadURL(
  location,
  json,
  rawJSONAvailable = process.env.NODE_ENV !== "production",
) {
  if (!rawJSONAvailable) {
    return `${JSON_DATA_URL_PREFIX}${encodeURIComponent(json)}`;
  }

  const pathname = location.pathname.replace(/\/$/, "");
  return `${location.origin}${pathname}/${JSON_DOWNLOAD_FILE_NAME}`;
}
