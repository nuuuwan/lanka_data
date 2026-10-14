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
const {
  makeSourceCatalog,
  renderSnapshot,
  writeDiscovery,
} = require("./buildDataDiscovery.cjs");

async function buildRawJSON() {
  await loadRegionData();
  const timestamp = new Date().toISOString();
  const catalogEntries = [];
  const baseURL =
    packageJson.homepage || "https://nuuuwan.github.io/lanka_data";
  const buildDir = path.join(__dirname, "../build");
  const sourceDefinitions = await Promise.all(
    DataSourceFactory.getDataSourceClasses().map(async (Source) => ({
      id: Source.name,
      baseURL: Source.getBaseURL(),
      metadataURL: Source.getMetadataURL(),
      metadata: await Source.getMetadata(),
    })),
  );
  const sourceCatalog = makeSourceCatalog(sourceDefinitions, timestamp);

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
    const entry = {
      query: visualQuery,
      description,
      url,
      pageURL: `${baseURL}/data/queries/${encodeURI(visualQuery)}/index.html`,
      visualURL: `${baseURL}/?/${encodeURI(visualQuery)}`,
      rowCount: datumSet.datumList.length,
      generatedAt: timestamp,
    };
    const tableDirectory = path.join(buildDir, "data/queries", visualQuery);
    await mkdir(tableDirectory, { recursive: true });
    await writeFile(
      path.join(tableDirectory, "index.html"),
      renderSnapshot(entry, datumSet, baseURL),
    );
    catalogEntries.push(entry);
  }

  const catalog = {
    catalogVersion: "1.1",
    description: "Catalog of published static JSON queries for Lanka Data",
    generatedAt: timestamp,
    count: catalogEntries.length,
    guideURL: `${baseURL}/data/index.html`,
    sourcesURL: `${baseURL}/sources.json`,
    queries: catalogEntries,
  };

  await mkdir(buildDir, { recursive: true });
  await writeFile(
    path.join(buildDir, "catalog.json"),
    JSON.stringify(catalog, null, 2),
  );
  await writeDiscovery({ buildDir, baseURL, catalog, sourceCatalog });
  console.log(
    "Published catalog.json, sources.json, API guide and readable data pages",
  );
}

buildRawJSON().catch((error) => {
  console.error("Could not build published raw JSON files:", error);
  process.exitCode = 1;
});
