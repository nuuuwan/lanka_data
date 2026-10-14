// Publish discovery instructions from local configuration only. Do not fetch
// metadata, query the data sources, or generate data as part of the build.
const { mkdir, writeFile, copyFile } = require("node:fs/promises");
const path = require("node:path");
const dataSources = require("../src/nonview/constants/DataSources.json");
const { homepage: baseURL } = require("../package.json");

const escapeHTML = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (char) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      })[char],
  );

function getSourceRegistry() {
  return {
    schemaVersion: "1.0",
    description:
      "Metadata endpoints used by the app. Fetch these live indexes to infer available datasets; this registry contains no observations or cached metadata.",
    sources: Object.entries(dataSources).map(([id, source]) => ({
      id,
      description: source.description,
      baseURL: source.baseURL,
      metadataURL: `${source.baseURL}/${source.metadataPath}`,
    })),
  };
}

function renderDirectory(registry) {
  const sources = registry.sources
    .map(
      (
        source,
      ) => `<li><strong>${escapeHTML(source.id)}</strong>: ${escapeHTML(source.description)}.<br>
<a href="${escapeHTML(source.metadataURL)}">Read live metadata</a><br>
Base URL for relative data paths: <code>${escapeHTML(source.baseURL)}</code></li>`,
    )
    .join("\n");
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Find data using live metadata | Lanka Data</title>
<style>body{font:16px/1.6 system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#182535}a{color:#1358a0}code{overflow-wrap:anywhere}li{margin:.8rem 0}</style>
</head><body><nav><a href="${baseURL}/">Lanka Data</a> · <a href="${baseURL}/api.md">API guide</a> · <a href="${baseURL}/llms.txt">For AI agents</a></nav>
<main><h1>Find Sri Lankan census and election data</h1>
<p>Discover available data from the same live metadata the app uses. This page requires no JavaScript. No authentication or API key is needed.</p>
<h2>Metadata sources</h2><ul>${sources}</ul>
<p><a href="${baseURL}/sources.json">Machine-readable source registry</a>. This lists metadata endpoints, not datasets or observations. The build publishes only local configuration and instructions; it does not download metadata or data.</p>
<h2>Retrieval procedure</h2><ol>
<li>Identify the subject, year, place, geographic level and breakdown requested.</li>
<li>Fetch the live metadata links above. Each index maps keys such as <code>House/Time+District+CookingFuel/Count</code> to relative data-file paths. Match the entity, set of dimension names and aggregate; order can vary. Do not assume separate dimensions can be combined into an unavailable cross-tabulation.</li>
<li>For each relevant path, append it to that source's base URL with a slash and fetch the file. Verify year and region in the contents; filenames and metadata keys alone do not establish every available value.</li>
<li>Read the nested JSON: entity → dimension:value → … → aggregate → typed value. For example, an integer count is encoded as <code>Int:123</code>. Select only the requested coordinates and preserve the metric's type and unit.</li>
<li>State the observation year, geography and unit. Cite the exact upstream data file and the official source. Missing rows are not zero; do not add overlapping geographic levels, years, totals or duplicate observations.</li></ol>
<p>Census attribution: <a href="https://www.statistics.gov.lk">Department of Census and Statistics of Sri Lanka</a>. Election attribution: <a href="https://elections.gov.lk">Elections Commission of Sri Lanka</a>. The files are served through the intermediary repositories used by the app. The API guide explains how to select the appropriate citation.</p>
<p>GitHub Pages hosts the app and these instructions; it does not execute queries or publish raw query results. Use upstream files directly, the interactive JSON download, or the local query endpoint. If retrieval fails or the metadata does not provide the requested combination, explain the gap instead of inventing a result.</p>
</main></body></html>\n`;
}

async function writeDiscovery(buildDir = path.join(__dirname, "../build")) {
  const registry = getSourceRegistry();
  await mkdir(path.join(buildDir, "data"), { recursive: true });
  await writeFile(
    path.join(buildDir, "data/index.html"),
    renderDirectory(registry),
  );
  await writeFile(
    path.join(buildDir, "sources.json"),
    JSON.stringify(registry, null, 2),
  );
  await copyFile(
    path.join(__dirname, "../api.md"),
    path.join(buildDir, "api.md"),
  );
}

if (require.main === module) {
  writeDiscovery().catch((error) => {
    console.error("Could not publish metadata discovery instructions:", error);
    process.exitCode = 1;
  });
}
module.exports = { getSourceRegistry, renderDirectory, writeDiscovery };
