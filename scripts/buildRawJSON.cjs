const { mkdir, writeFile } = require("node:fs/promises");
const path = require("node:path");
const { createJiti } = require("jiti");

const jiti = createJiti(__filename);
const { PUBLISHED_RAW_JSON_QUERIES } = jiti(
  "../src/nonview/constants/RawJSON.js",
);
const { JSON_DOWNLOAD_FILE_NAME } = jiti("../src/nonview/constants/APP.js");
const { loadRegionData } = jiti(
  "../src/nonview/core/raw_json/RawJSONMiddleware.js",
);
const Query = jiti("../src/nonview/core/Query.js").default;
const getQueryFinding = jiti("../src/nonview/core/QueryFinding.js").default;
const DataSourceFactory = jiti(
  "../src/nonview/core/data_source/DataSourceFactory.js",
).default;
const packageJson = require("../package.json");

async function buildRawJSON() {
  await loadRegionData();
  const timestamp = new Date().toISOString();
  const catalogEntries = [];
  const baseURL = packageJson.homepage || "https://nuuuwan.github.io/lanka_data";

  for (const visualQuery of PUBLISHED_RAW_JSON_QUERIES) {
    const queryString = visualQuery.slice(0, -"/JSON".length);
    const query = await Query.fromString(queryString);
    const datumSet = await DataSourceFactory.getDatumSetForQuery(query);
    if (datumSet.datumList.length === 0) {
      throw new Error(`No data returned for published query: ${visualQuery}`);
    }
    const directory = path.join(__dirname, "../build", visualQuery);
    await mkdir(directory, { recursive: true });
    await writeFile(
      path.join(directory, JSON_DOWNLOAD_FILE_NAME),
      JSON.stringify(datumSet, null, 2),
    );
    console.log(`Published ${visualQuery}/${JSON_DOWNLOAD_FILE_NAME}`);

    const description = getQueryFinding(query);
    const url = `${baseURL}/${encodeURI(visualQuery)}/${JSON_DOWNLOAD_FILE_NAME}`;
    catalogEntries.push({
      query: visualQuery,
      description,
      url,
      generatedAt: timestamp,
    });
  }

  const catalog = {
    catalogVersion: "1.0",
    description: "Catalog of published static JSON queries for Lanka Data",
    generatedAt: timestamp,
    count: catalogEntries.length,
    queries: catalogEntries,
  };

  const buildDir = path.join(__dirname, "../build");
  await mkdir(buildDir, { recursive: true });
  await writeFile(
    path.join(buildDir, "catalog.json"),
    JSON.stringify(catalog, null, 2),
  );
  await writeFile(
    path.join(__dirname, "../public/catalog.json"),
    JSON.stringify(catalog, null, 2),
  );
  console.log("Published catalog.json");
}

buildRawJSON().catch((error) => {
  console.error("Could not build published raw JSON files:", error);
  process.exitCode = 1;
});
