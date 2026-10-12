import {
  APP_PATH,
  JSON_DATA_URL_PREFIX,
  JSON_DOWNLOAD_FILE_NAME,
} from "../../constants/APP.js";
import { PUBLISHED_RAW_JSON_QUERIES } from "../../constants/RawJSON.js";

export function getJSONDownloadURL(
  location,
  json,
  rawJSONAvailable = process.env.NODE_ENV !== "production",
) {
  const pathname = location.pathname.replace(/\/$/, "");
  const isPublished = PUBLISHED_RAW_JSON_QUERIES.some(
    (query) => decodeURIComponent(pathname) === `${APP_PATH}/${query}`,
  );
  if (!rawJSONAvailable && !isPublished) {
    return `${JSON_DATA_URL_PREFIX}${encodeURIComponent(json)}`;
  }

  return `${location.origin}${pathname}/${JSON_DOWNLOAD_FILE_NAME}`;
}
