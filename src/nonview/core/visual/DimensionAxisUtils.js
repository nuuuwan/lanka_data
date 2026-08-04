import Region from "../thing/concept/category_concept/region/region/Region.js";

export function getDimIndexInfo(datumList) {
  const nDims = datumList[0].query.dimThingList.length;
  const varyingDimIndexes = [];
  for (let dimIndex = 0; dimIndex < nDims; dimIndex++) {
    const values = new Set(
      datumList.map((datum) => datum.query.dimThingList[dimIndex].value),
    );
    if (values.size > 1) {
      varyingDimIndexes.push(dimIndex);
    }
  }
  return { nDims, varyingDimIndexes };
}

export function isRegionDim(datumList, dimIndex) {
  return datumList[0].query.dimThingList[dimIndex] instanceof Region;
}

export function getXAxisDimIndex(datumList, stackDimIndex = null) {
  const { varyingDimIndexes } = getDimIndexInfo(datumList);
  if (varyingDimIndexes.length === 0) {
    return 0;
  }
  if (stackDimIndex === null || varyingDimIndexes.length === 1) {
    return varyingDimIndexes.at(-1);
  }
  return varyingDimIndexes.at(-2);
}

export function getStackDimIndex(datumList) {
  const { varyingDimIndexes } = getDimIndexInfo(datumList);
  return varyingDimIndexes.length < 2 ? null : varyingDimIndexes.at(-1);
}

export function getMarimekkoDimIndexes(datumList) {
  const { varyingDimIndexes } = getDimIndexInfo(datumList);
  if (varyingDimIndexes.length === 0) {
    return { xAxisDimIndex: 0, stackDimIndex: null };
  }
  if (varyingDimIndexes.length === 1) {
    return { xAxisDimIndex: varyingDimIndexes[0], stackDimIndex: null };
  }
  const regionDimIndex = varyingDimIndexes.find((dimIndex) =>
    isRegionDim(datumList, dimIndex),
  );
  if (regionDimIndex !== undefined) {
    const stackDimIndex = varyingDimIndexes.at(-1);
    return stackDimIndex === regionDimIndex
      ? {
          xAxisDimIndex: regionDimIndex,
          stackDimIndex: varyingDimIndexes.at(-2),
        }
      : { xAxisDimIndex: regionDimIndex, stackDimIndex };
  }
  return {
    xAxisDimIndex: varyingDimIndexes.at(-2),
    stackDimIndex: varyingDimIndexes.at(-1),
  };
}
