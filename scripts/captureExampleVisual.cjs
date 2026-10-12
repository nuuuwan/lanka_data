const { readFile } = require("node:fs/promises");
const path = require("node:path");

module.exports = async function captureExampleVisual(
  page,
  url,
  directory,
  image,
) {
  await page.goto(url);
  await page.waitForFunction(() => {
    const error = document.querySelector('[data-testid="query-error"]');
    if (error) throw new Error(error.textContent);
    const button = document.querySelector(
      'button[aria-label="Download visual as PNG"]',
    );
    const visual = document.querySelector('[data-testid="fitted-visual"]');
    return (
      button &&
      !button.disabled &&
      visual &&
      !visual.querySelector('[role="progressbar"]') &&
      visual.querySelector(
        "svg path, svg rect, svg circle, svg polygon, canvas, table",
      )
    );
  });
  await page
    .locator('[data-testid="fitted-visual-section"] header h1')
    .waitFor();
  await page.locator('[data-testid="fitted-visual-section"] aside').waitFor();
  await page.evaluate(async () => {
    await document.fonts.ready;
    const section = document.querySelector(
      '[data-testid="fitted-visual-section"]',
    );
    const content = section.querySelector('[data-testid="fitted-visual"]')
      .firstElementChild.firstElementChild;
    Object.assign(section.style, {
      width: "1000px",
      left: "0",
      transform: "none",
    });
    Object.assign(content.style, {
      position: "relative",
      left: "0",
      transform: "none",
    });
    content.parentElement.style.height = "auto";
    await new Promise((resolve) =>
      requestAnimationFrame(() => requestAnimationFrame(resolve)),
    );
  });
  const pendingDownload = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download visual as PNG" }).click();
  const download = await pendingDownload;
  if (download.suggestedFilename() !== image) {
    throw new Error(`Unexpected download filename for ${url}`);
  }
  await download.saveAs(path.join(directory, image));
  const png = await readFile(path.join(directory, image));
  if (png.length < 24 || png.readUInt32BE(0) !== 0x89504e47) {
    throw new Error(`Invalid PNG: ${image}`);
  }
};
