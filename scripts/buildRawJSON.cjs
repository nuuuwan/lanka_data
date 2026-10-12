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
const DataSourceFactory = jiti(
  "../src/nonview/core/data_source/DataSourceFactory.js",
).default;

async function buildRawJSON() {
  await loadRegionData();
  for (const visualQuery of PUBLISHED_RAW_JSON_QUERIES) {
    const queryString = visualQuery.slice(0, -"/JSON".length);
    const datumSet = await DataSourceFactory.getDatumSetForQuery(
      await Query.fromString(queryString),
    );
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
  }
}

buildRawJSON().catch((error) => {
  console.error("Could not build published raw JSON files:", error);
  process.exitCode = 1;
});
