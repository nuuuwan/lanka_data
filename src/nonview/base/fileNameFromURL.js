const MAX_FILE_NAME_LENGTH = 200;

export default function fileNameFromURL(url, extension, basePath = "") {
  const parsedURL = new URL(url);
  const pathname = parsedURL.pathname.startsWith(`${basePath}/`)
    ? parsedURL.pathname.slice(basePath.length)
    : parsedURL.pathname;
  const name = decodeURIComponent(
    `${pathname}${parsedURL.search}${parsedURL.hash}`,
  )
    .replace(/[^a-zA-Z0-9._-]+/g, "-")
    .replace(/^[.-]+|[.-]+$/g, "");
  const suffix = `.${extension}`;
  return `${name.slice(0, MAX_FILE_NAME_LENGTH - suffix.length).replace(/[.-]+$/g, "")}${suffix}`;
}
