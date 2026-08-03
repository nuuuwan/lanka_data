import Query from "../Query.js";
import DataSourceFactory from "../data_source/DataSourceFactory.js";
import RegionFactory from "../thing/concept/category_concept/region/RegionFactory.js";
import {
  APP_PATH,
  JSON_DOWNLOAD_FILE_NAME,
} from "../../constants/APP.js";

const RAW_JSON_SUFFIX = `/JSON/${JSON_DOWNLOAD_FILE_NAME}`;

async function loadRegionData() {
  await Promise.all(
    RegionFactory.list().map(async (RegionClass) => {
      RegionClass.ents = await RegionClass.loadEnts();
    }),
  );
}

export function getQueryStringFromRawJSONURL(url) {
  let pathname;
  try {
    pathname = decodeURIComponent(new URL(url, "http://localhost").pathname);
  } catch {
    return null;
  }

  if (!pathname.startsWith(`${APP_PATH}/`) || !pathname.endsWith(RAW_JSON_SUFFIX)) {
    return null;
  }

  return pathname.slice(APP_PATH.length + 1, -RAW_JSON_SUFFIX.length);
}

export function createRawJSONMiddleware({
  ensureRegionData = loadRegionData,
  getDatumSet = (query) => DataSourceFactory.getDatumSetForQuery(query),
  parseQuery = (queryString) => Query.fromString(queryString),
} = {}) {
  let regionDataPromise;

  return async function rawJSONMiddleware(request, response, next) {
    const queryString = getQueryStringFromRawJSONURL(
      request.originalUrl || request.url,
    );
    if (queryString === null) {
      next();
      return;
    }

    try {
      regionDataPromise ??= ensureRegionData().catch((error) => {
        regionDataPromise = null;
        throw error;
      });
      await regionDataPromise;
      const query = await parseQuery(queryString);
      const datumSet = await getDatumSet(query);
      response.status(200);
      response.set("Content-Type", "application/json; charset=utf-8");
      response.set(
        "Content-Disposition",
        `attachment; filename="${JSON_DOWNLOAD_FILE_NAME}"`,
      );
      response.send(JSON.stringify(datumSet, null, 2));
    } catch (error) {
      console.error("[RawJSONMiddleware] Could not load requested data", error);
      response.status(500).json({
        error: "We couldn't load the requested data.",
      });
    }
  };
}
