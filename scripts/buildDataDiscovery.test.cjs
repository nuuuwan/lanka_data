const test = require("node:test");
const assert = require("node:assert/strict");
const { mkdtemp, readFile, rm } = require("node:fs/promises");
const os = require("node:os");
const path = require("node:path");
const {
  makeSourceCatalog,
  renderSnapshot,
  renderSources,
  writeDiscovery,
} = require("./buildDataDiscovery.cjs");

const baseURL = "https://example.org/lanka_data";
const generatedAt = "2026-10-14T00:00:00.000Z";
const entry = {
  query: "Vote/Time=2024+PD<ED=colombo+Party/Count/JSON",
  description: "Votes <by party>",
  url: `${baseURL}/Vote/Time=2024+PD%3CED=colombo+Party/Count/JSON/raw.json`,
  pageURL: `${baseURL}/data/queries/votes/index.html`,
  visualURL: `${baseURL}/?/Vote/Time=2024+PD%3CED=colombo+Party/Count/JSON`,
  rowCount: 2,
  generatedAt,
};
const sourceCatalog = makeSourceCatalog(
  [
    {
      id: "Source",
      baseURL: "https://example.org/repo",
      metadataURL: "https://example.org/repo/metadata.json",
      metadata: {
        "Person/Time+District/Count": ["population.json"],
        "Person/Time+Province/Count": ["population.json", "older.json"],
      },
    },
  ],
  generatedAt,
);

test("source catalog preserves every metadata key and expands all source paths", () => {
  assert.deepEqual(sourceCatalog.sources[0].datasets, [
    {
      metadataKey: "Person/Time+District/Count",
      urls: ["https://example.org/repo/population.json"],
    },
    {
      metadataKey: "Person/Time+Province/Count",
      urls: [
        "https://example.org/repo/population.json",
        "https://example.org/repo/older.json",
      ],
    },
  ]);
  const html = renderSources(sourceCatalog, baseURL);
  assert.equal(
    html.match(/href="https:\/\/example.org\/repo\/population.json"/g).length,
    1,
  );
  for (const dataset of sourceCatalog.sources[0].datasets)
    assert.ok(html.includes(dataset.metadataKey));
});

test("HTML preserves exact row coordinates, zero values and provenance without executing markup", () => {
  const datumSet = {
    datumList: [
      {
        query: { queryStr: "Vote/Party=NPP+PD=colombo_north+Time=2024/Count" },
        answerThing: { value: "17978" },
      },
      {
        query: {
          queryStr: "Vote/Party=<script>+PD=colombo_north+Time=2024/Count",
        },
        answerThing: { value: "0" },
      },
    ],
    provenance: [
      {
        source: 'Agency & "name"',
        title: "Election results",
        url: "https://example.org/?a=1&b=2",
      },
    ],
  };
  const html = renderSnapshot(entry, datumSet, baseURL);
  assert.ok(html.includes("Vote/Party=NPP+PD=colombo_north+Time=2024/Count"));
  assert.ok(html.includes("<td>17978</td>"));
  assert.ok(html.includes("<td>0</td>"));
  assert.ok(html.includes("Party=&lt;script&gt;"));
  assert.ok(!html.includes("<script>"));
  assert.ok(html.includes("Agency &amp; &quot;name&quot;"));
  assert.ok(html.includes('href="https://example.org/?a=1&amp;b=2"'));
  assert.ok(html.includes(entry.url));
});

test("generated directory connects snapshots, upstream files and the deployed guide", async () => {
  const buildDir = await mkdtemp(path.join(os.tmpdir(), "lanka-discovery-"));
  try {
    await writeDiscovery({
      buildDir,
      baseURL,
      catalog: { generatedAt, queries: [entry] },
      sourceCatalog,
    });
    const html = await readFile(path.join(buildDir, "data/index.html"), "utf8");
    for (const url of [
      entry.url,
      entry.pageURL,
      entry.visualURL,
      `${baseURL}/sources.json`,
      `${baseURL}/data/sources.html`,
      `${baseURL}/api.md`,
    ])
      assert.ok(html.includes(url));
    assert.deepEqual(
      JSON.parse(await readFile(path.join(buildDir, "sources.json"), "utf8")),
      sourceCatalog,
    );
    assert.equal(
      await readFile(path.join(buildDir, "api.md"), "utf8"),
      await readFile(path.join(__dirname, "../api.md"), "utf8"),
    );
  } finally {
    await rm(buildDir, { recursive: true });
  }
});
