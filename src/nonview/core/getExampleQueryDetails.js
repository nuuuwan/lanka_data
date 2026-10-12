import fileNameFromURL from "../base/fileNameFromURL.js";
import { APP_PATH } from "../constants/APP.js";

function humanize(value) {
  return value
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replaceAll("_", " ")
    .replace(/\bED\b/g, "electoral district")
    .replace(/\bPD\b/g, "polling division")
    .toLowerCase();
}

export default function getExampleQueryDetails(query) {
  const [entity, facets, aggregate, visual] = query.split("/");
  const dimensions = facets.split("+");
  const year = dimensions.find((facet) => facet.startsWith("Time="))?.slice(5);
  const election = dimensions
    .find((facet) => facet.startsWith("ElectionType="))
    ?.slice(13);
  const subject =
    aggregate === "Count" && entity === "Person"
      ? "population"
      : `${humanize(entity)} ${humanize(aggregate)}`;
  const details = dimensions
    .filter((facet) => !/^(Time|ElectionType)=/.test(facet))
    .map((facet) =>
      humanize(facet).replaceAll("=", ": ").replaceAll("<", " in "),
    )
    .join(", ");
  const title = [year, election && humanize(election), subject]
    .filter(Boolean)
    .join(" ");
  const label = `${title}: ${details} (${humanize(visual)})`;
  return {
    label: label.charAt(0).toUpperCase() + label.slice(1),
    image: fileNameFromURL(
      `https://example.invalid${APP_PATH}/${query}`,
      "png",
      APP_PATH,
    ),
  };
}
