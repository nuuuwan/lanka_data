const test = require("node:test");
const assert = require("node:assert/strict");
const { mkdtemp, readFile, readdir, rm } = require("node:fs/promises");
const os = require("node:os");
const path = require("node:path");
const { createJiti } = require("jiti");
const {
  getSourceRegistry,
  renderDirectory,
  writeDiscovery,
} = require("./buildDataDiscovery.cjs");
const jiti = createJiti(__filename);
const DataSourceFactory = jiti(
  "../src/nonview/core/data_source/DataSourceFactory.js",
).default;
const { getJSONDownloadURL } = jiti("../src/nonview/core/raw_json/RawJSON.js");

test("discovery endpoints match every data source used by the app", () => {
  const registry = getSourceRegistry();
  const classes = DataSourceFactory.getDataSourceClasses();
  assert.deepEqual(
    registry.sources.map((s) => s.id).sort(),
    classes.map((s) => s.name).sort(),
  );
  for (const Source of classes) {
    const source = registry.sources.find((s) => s.id === Source.name);
    assert.equal(source.baseURL, Source.getBaseURL());
    assert.equal(source.metadataURL, Source.getMetadataURL());
  }
});

test("discovery build performs no fetch and publishes only configuration and instructions", async () => {
  const buildDir = await mkdtemp(path.join(os.tmpdir(), "lanka-discovery-"));
  const originalFetch = global.fetch;
  global.fetch = () => {
    throw new Error("Build must not fetch metadata or data");
  };
  try {
    await writeDiscovery(buildDir);
    assert.deepEqual((await readdir(buildDir)).sort(), [
      "api.md",
      "data",
      "sources.json",
    ]);
    assert.deepEqual(await readdir(path.join(buildDir, "data")), [
      "index.html",
    ]);
    const registry = JSON.parse(
      await readFile(path.join(buildDir, "sources.json"), "utf8"),
    );
    assert.deepEqual(registry, getSourceRegistry());
    const html = await readFile(path.join(buildDir, "data/index.html"), "utf8");
    for (const source of registry.sources) {
      assert.ok(html.includes(`href="${source.metadataURL}"`));
      assert.ok(html.includes(source.baseURL));
      assert.equal(source.datasets, undefined);
    }
    assert.equal(
      await readFile(path.join(buildDir, "api.md"), "utf8"),
      await readFile(path.join(__dirname, "../api.md"), "utf8"),
    );
  } finally {
    global.fetch = originalFetch;
    await rm(buildDir, { recursive: true });
  }
});

test("directory safely renders source metadata links and descriptions", () => {
  const html = renderDirectory({
    sources: [
      {
        id: "<script>",
        description: "A & B",
        metadataURL: 'https://example.org/?a=1&b="2"',
        baseURL: "https://example.org",
      },
    ],
  });
  assert.ok(html.includes("&lt;script&gt;"));
  assert.ok(html.includes("A &amp; B"));
  assert.ok(html.includes("?a=1&amp;b=&quot;2&quot;"));
});

test("production downloads use fetched JSON; local downloads retain on-demand endpoints", () => {
  const location = {
    origin: "https://example.org",
    pathname:
      "/lanka_data/Vote/ElectionType=presidential+Time=2024+PD%3CED=colombo+Party/Count/JSON/",
  };
  const json = '{"datumList":[],"provenance":[]}';
  assert.equal(
    getJSONDownloadURL(location, json, false),
    `data:application/json;charset=utf-8,${encodeURIComponent(json)}`,
  );
  assert.equal(
    getJSONDownloadURL(location, json, true),
    `${location.origin}${location.pathname}raw.json`,
  );
});
