import String from "../../../../../../base/String.js";

export default class RegionMatcherMixin {
  static getChildRegions(parentRegion, ChildRegionClass) {
    const parentEnt = parentRegion.getEnt();
    const childRegionEnts = ChildRegionClass.ents;

    const parentId = parentEnt.id;
    const parentIdKey = `${parentRegion.constructor.regionClassId()}_id`;

    const matchingChildValues = [];
    for (const childRegionEnt of childRegionEnts) {
      const childParentId = childRegionEnt[parentIdKey];
      if (childParentId === parentId) {
        matchingChildValues.push(String.toSnakeCase(childRegionEnt.name));
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
