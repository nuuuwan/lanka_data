export function getMetadataKeyFromParts(entityClass, dimThingList, aggregate) {
  const dimensionToken = dimThingList
    .map((dimThing) => dimThing.constructor.getClassName())
    .join("+");
  return [entityClass.getClassName(), dimensionToken, aggregate].join("/");
}

export function normalizeMetadataKey(metadataKey) {
  const [entityClassName, dimensionToken, aggregate] = metadataKey.split("/");
  const normalizedDimensionToken = dimensionToken
    .split("+")
    .filter(Boolean)
    .sort()
    .join("+");
  return [entityClassName, normalizedDimensionToken, aggregate].join("/");
}
