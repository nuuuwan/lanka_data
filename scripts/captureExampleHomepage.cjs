const path = require("node:path");

module.exports = async function captureExampleHomepage(
  page,
  baseURL,
  directory,
  names,
  queries,
) {
  await page.route("**/example-images/*.png", async (route) => {
    const name = decodeURIComponent(
      new URL(route.request().url()).pathname.split("/").at(-1),
    );
    if (!names.has(name) || name === "homepage.png") return route.continue();
    await route.fulfill({
      path: path.join(directory, name),
      contentType: "image/png",
    });
  });
  await page.goto(baseURL.href);
  await page.locator("li img").first().waitFor();
  await page.evaluate(() => {
    document.querySelectorAll("li img").forEach((image) => {
      image.loading = "eager";
    });
  });
  await page.waitForFunction(() =>
    [...document.querySelectorAll("li img")].every(
      (image) => image.complete && image.naturalWidth > 0,
    ),
  );
  const links = await page
    .locator("li a")
    .evaluateAll((elements) =>
      elements.map((link) => decodeURI(link.getAttribute("href"))),
    );
  if (
    JSON.stringify(links) !==
    JSON.stringify(queries.map((query) => `${baseURL.pathname}${query}`))
  ) {
    throw new Error(
      "The running app's example list differs from EXAMPLE_QUERIES",
    );
  }
  await page.evaluate(() => document.fonts.ready);
  await page.screenshot({
    path: path.join(directory, "homepage.png"),
    fullPage: true,
    animations: "disabled",
  });
};
