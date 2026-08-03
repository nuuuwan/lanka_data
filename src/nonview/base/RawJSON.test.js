import { JSON_DATA_URL_PREFIX } from "../constants/APP.js";
import { getJSONDownloadURL } from "./RawJSON.js";

test("uses a data URL when the raw endpoint is unavailable", () => {
  const json = '{"count":42}';
  const location = new URL("https://nuuuwan.github.io/lanka_data/query/JSON");

  expect(getJSONDownloadURL(location, json, false)).toBe(
    `${JSON_DATA_URL_PREFIX}${encodeURIComponent(json)}`,
  );
});
