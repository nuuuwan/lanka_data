const { mkdir, writeFile, copyFile } = require("node:fs/promises");
const path = require("node:path");

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
const link = (url, label) =>
  `<a href="${escapeHTML(url)}">${escapeHTML(label)}</a>`;

function page(title, content, baseURL) {
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escapeHTML(title)} | Lanka Data</title>
<style>body{font:16px/1.6 system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#182535}a{color:#1358a0}table{border-collapse:collapse;width:100%}th,td{padding:.5rem;text-align:left;border:1px solid #ccd3db;vertical-align:top}code{overflow-wrap:anywhere}li{margin:.6rem 0}.table{overflow-x:auto}</style>
</head><body><nav>${link(`${baseURL}/`, "Lanka Data")} · ${link(`${baseURL}/data/index.html`, "Data directory")} · ${link(`${baseURL}/data/sources.html`, "Upstream datasets")} · ${link(`${baseURL}/api.md`, "API guide")}</nav>
<main><h1>${escapeHTML(title)}</h1>${content}</main></body></html>\n`;
}

function renderSnapshot(entry, datumSet, baseURL) {
  // The response's per-row query names its dimensions. Their order can differ
  // from the requested query, so do not zip values with the request's dimensions.
  const rows = datumSet.datumList
    .map(
      ({ query, answerThing }) =>
        `<tr><td><code>${escapeHTML(query.queryStr)}</code></td><td>${escapeHTML(answerThing.value)}</td></tr>`,
    )
    .join("\n");
  const citations = datumSet.provenance
    .map(
      ({ source, title, url }) =>
        `<li>${link(url, title)} — ${escapeHTML(source)}</li>`,
    )
    .join("\n");
  return page(
    entry.description,
    `<p>Query: <code>${escapeHTML(entry.query)}</code></p>
<p>Snapshot built ${escapeHTML(entry.generatedAt)}. This is the build time, not the observation date. Use each row's Time dimension.</p>
<p>${link(entry.url, "Download JSON")} · ${link(entry.visualURL, "Open interactive view")}</p>
<p>${entry.rowCount} observations. Values are exact stored values; missing rows do not establish zero. Count means people for Person, households for the household examples, and votes for Vote. Do not add different years or overlapping geographies.</p>
<h2>Sources</h2><ul>${citations}</ul>
<h2>Data</h2><div class="table"><table><thead><tr><th scope="col">Observation query (dimension names and values)</th><th scope="col">Value</th></tr></thead><tbody>${rows}</tbody></table></div>`,
    baseURL,
  );
}

function makeSourceCatalog(sourceDefinitions, generatedAt) {
  return {
    catalogVersion: "1.0",
    generatedAt,
    description:
      "Upstream metadata indexed at build time. Linked source files remain live and use nested Lanka Data JSON, not datumList responses.",
    sources: sourceDefinitions.map(
      ({ id, baseURL, metadataURL, metadata }) => ({
        id,
        baseURL,
        metadataURL,
        datasets: Object.entries(metadata).map(([metadataKey, paths]) => ({
          metadataKey,
          urls: paths.map((partialPath) => `${baseURL}/${partialPath}`),
        })),
      }),
    ),
  };
}

function renderSources(sourceCatalog, baseURL) {
  const sections = sourceCatalog.sources
    .map((source) => {
      // List each physical file once, with every supported dimension combination.
      const files = new Map();
      for (const { metadataKey, urls } of source.datasets) {
        for (const url of urls) {
          if (!files.has(url)) files.set(url, []);
          files.get(url).push(metadataKey);
        }
      }
      return `<h2>${escapeHTML(source.id)}</h2><p>${link(source.metadataURL, "Live metadata index")} — ${files.size} source files</p>
<ul>${[...files].map(([url, keys]) => `<li>${link(url, decodeURI(url.slice(source.baseURL.length + 1)))}<br>Dimension combinations: ${keys.map((key) => `<code>${escapeHTML(key)}</code>`).join("; ")}</li>`).join("\n")}</ul>`;
    })
    .join("\n");
  return page(
    "Complete upstream dataset directory",
    `<p>This directory indexes all metadata entries from the app's data sources, including tables beyond the published query snapshots. Indexed ${escapeHTML(sourceCatalog.generatedAt)}. This does not guarantee every upstream table is supported by the interactive query engine.</p>
<p>${link(`${baseURL}/sources.json`, "Machine-readable source directory")}. Follow the exact file links below; confirm the year, dimensions, units and values in the file itself. File names alone do not establish coverage.</p>
<p>These files contain nested objects: entity → dimension:value → … → aggregate → typed value, for example <code>House → Time:2024 → Country:sri_lanka → CookingFuel:fire_wood → Count → Int:3381781</code>. They do not have a <code>datumList</code> field. A file can contain several geographic levels; select one before calculating totals. See the API guide for traversal code and citation rules.</p>
${sections}`,
    baseURL,
  );
}

async function writeDiscovery({ buildDir, baseURL, catalog, sourceCatalog }) {
  await mkdir(path.join(buildDir, "data"), { recursive: true });
  const entries = catalog.queries
    .map(
      (entry) =>
        `<li><strong>${link(entry.pageURL, entry.description)}</strong><br><code>${escapeHTML(entry.query)}</code><br>${link(entry.url, "JSON")} · ${link(entry.visualURL, "Interactive view")} · ${entry.rowCount} observations</li>`,
    )
    .join("\n");
  const index = page(
    "Find Sri Lankan census and election data",
    `<p>Lanka Data brings census and election data into one query tool. This directory and the snapshot tables work without JavaScript, authentication or an API key.</p>
<h2>How to answer a data question</h2><ol>
<li>Identify the subject (people, households or votes), year, geography and breakdown requested. Check these against a snapshot below.</li>
<li>Open its readable table or JSON link and inspect the actual observations. Filter to the requested coordinates. Cite the table or JSON URL and its official provenance.</li>
<li>If no snapshot fits, search the ${link(`${baseURL}/data/sources.html`, "complete upstream dataset directory")} for the topic and dimension combination, then fetch the linked source file. Verify coverage in the file before answering.</li>
<li>If the data or a required browsing capability is unavailable, say what could not be retrieved. Do not infer missing values or substitute another year or geography.</li></ol>
<p>Sources are the Department of Census and Statistics of Sri Lanka and the Elections Commission of Sri Lanka, accessed through the public repositories used by the app. Availability varies by topic, year and geographic level.</p>
<p>${link(`${baseURL}/catalog.json`, "Snapshot catalog (JSON)")} · ${link(`${baseURL}/sources.json`, "Upstream directory (JSON)")} · ${link(`${baseURL}/llms.txt`, "AI agent entry point")} · ${link(`${baseURL}/api.md`, "Full schema, query syntax and retrieval examples")}</p>
<h2>Published query snapshots</h2><p>Built ${escapeHTML(catalog.generatedAt)}. Build dates describe snapshots, not census or election dates. All homepage examples are included. GitHub Pages serves only these prebuilt queries; arbitrary raw.json paths do not execute queries.</p><ul>${entries}</ul>`,
    baseURL,
  );
  await Promise.all([
    writeFile(path.join(buildDir, "data/index.html"), index),
    writeFile(
      path.join(buildDir, "data/sources.html"),
      renderSources(sourceCatalog, baseURL),
    ),
    writeFile(
      path.join(buildDir, "sources.json"),
      JSON.stringify(sourceCatalog, null, 2),
    ),
    copyFile(path.join(__dirname, "../api.md"), path.join(buildDir, "api.md")),
  ]);
}

module.exports = {
  makeSourceCatalog,
  renderSnapshot,
  renderSources,
  writeDiscovery,
};
