const { mkdtemp, mkdir, copyFile, rm } = require("node:fs/promises");
const path = require("node:path");
const os = require("node:os");
const { chromium } = require("playwright");
const { createJiti } = require("jiti");
const captureExampleVisual = require("./captureExampleVisual.cjs");
const captureExampleHomepage = require("./captureExampleHomepage.cjs");

const jiti = createJiti(__filename);
const { EXAMPLE_QUERIES } = jiti("../src/nonview/constants/ExampleQueries.js");
const { APP_PATH } = jiti("../src/nonview/constants/APP.js");
const getExampleQueryDetails = jiti(
  "../src/nonview/core/getExampleQueryDetails.js",
).default;
const output = path.join(__dirname, "../public/example-images");
const baseURL = new URL(
  APP_PATH + "/",
  process.argv[2] || "http://localhost:3000",
);

async function buildExampleImages() {
  const staging = await mkdtemp(path.join(os.tmpdir(), "lanka-data-previews-"));
  let browser;
  try {
    browser = await chromium.launch();
    const page = await browser.newPage({
      viewport: { width: 1440, height: 1400 },
    });
    page.setDefaultTimeout(120_000);
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("console", (message) => {
      if (message.type() === "error") errors.push(message.text());
    });
    const names = new Set(["homepage.png"]);
    for (const query of EXAMPLE_QUERIES) {
      const { image } = getExampleQueryDetails(query);
      if (names.has(image))
        throw new Error(`Duplicate preview filename: ${image}`);
      names.add(image);
      await captureExampleVisual(
        page,
        new URL(query, baseURL).href,
        staging,
        image,
      );
      if (errors.length) throw new Error(errors.join("\n"));
      console.log(`Captured ${image}`);
    }
    await captureExampleHomepage(
      page,
      baseURL,
      staging,
      names,
      EXAMPLE_QUERIES,
    );
    if (errors.length) throw new Error(errors.join("\n"));
    await mkdir(output, { recursive: true });
    for (const name of names)
      await copyFile(path.join(staging, name), path.join(output, name));
    console.log(
      `Saved ${EXAMPLE_QUERIES.length} previews and homepage.png to ${output}`,
    );
  } finally {
    try {
      if (browser) await browser.close();
    } finally {
      await rm(staging, { recursive: true });
    }
  }
}

buildExampleImages().catch((error) => {
  console.error("Could not rebuild example screenshots:", error);
  process.exitCode = 1;
});
