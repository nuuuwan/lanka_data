import { QUERY_DELIMITERS } from "./QueryTokens.js";

export function getMetadataKey(entityClass, dimensions, aggregate) {
  const dimensionToken = dimensions
    .map((thing) => thing.constructor.getClassName())
    .join(QUERY_DELIMITERS.dimension);
  return [entityClass.getClassName(), dimensionToken, aggregate].join(
    QUERY_DELIMITERS.token,
  );
}

export function normalizeMetadataKey(metadataKey) {
  const [entityName, dimensionToken, aggregate] = metadataKey.split(
    QUERY_DELIMITERS.token,
  );
  const normalizedDimensions = dimensionToken
    .split(QUERY_DELIMITERS.dimension)
    .filter(Boolean)
    .sort()
    .join(QUERY_DELIMITERS.dimension);
  return [entityName, normalizedDimensions, aggregate].join(
    QUERY_DELIMITERS.token,
  );
}
