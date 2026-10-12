import { toBlob } from "html-to-image";

export default async function downloadElementPNG(
  element,
  fileName,
  pixelRatio,
) {
  if (!element) throw new Error("The visual is not ready to download.");
  await document.fonts.ready;
  const blob = await toBlob(element, {
    backgroundColor: "#ffffff",
    pixelRatio,
    skipFonts: true,
    style: { position: "static", transform: "none", left: "0" },
  });
  if (!blob) throw new Error("Could not create the PNG image.");
  const url = URL.createObjectURL(blob);
  try {
    const link = document.createElement("a");
    link.href = url;
    link.download = fileName;
    link.click();
  } finally {
    URL.revokeObjectURL(url);
  }
}
