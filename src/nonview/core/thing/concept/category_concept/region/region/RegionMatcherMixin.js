function toSnakeCase(value) {
  return String(value)
    .trim()
    .replace(/&/g, " and ")
    .replace(/[()]/g, " ")
    .replace(/([a-z])([A-Z])/g, "$1_$2")
    .replace(/\s+/g, "_")
    .replace(/[^a-zA-Z0-9_]+/g, "_")
    .replace(/_+/g, "_")
    .toLowerCase();
}

export default class RegionMatcherMixin {
  static getChildRegions(parentRegion, ChildRegionClass) {
    const parentEnt = parentRegion.getEnt();
    const childRegionEnts = ChildRegionClass.getEnts();

    const parentId = parentEnt.id;
    const parentIdKey = `${parentRegion.constructor.regionClassId()}_id`;

    const matchingChildValues = [];
    for (const childRegionEnt of childRegionEnts) {
      const childParentId = childRegionEnt[parentIdKey];
      if (childParentId === parentId) {
        matchingChildValues.push(toSnakeCase(childRegionEnt.name));
      }
    }

    if (matchingChildValues.length === 0) {
      throw new Error(
        `No child regions of type ${ChildRegionClass.name}` +
          ` found for parent region: ${parentRegion.getHumanReadableValue()}`,
      );
    }

    return matchingChildValues.map((value) => new ChildRegionClass(value));
  }
}
