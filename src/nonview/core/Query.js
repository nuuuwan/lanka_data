import {
  getParentRegionFilter,
  getSubRegionFilter,
} from "./query/QueryFilters.js";
import {
  getMetadataKeyFromParts,
  normalizeMetadataKey,
} from "./query/QueryMetadataUtils.js";
import { parseKeyValueList, parseQueryString } from "./query/QueryParser.js";
import isQuerySubset from "./query/QuerySubset.js";
import {
  expandSubRegionDimThingList,
  getQueryStringFromParts,
  getThingFromToken,
  getThingValues,
} from "./query/QueryTokenUtils.js";

export default class Query {
  static DELIM_TOKEN = "/";
  static DELIM_DIM = "+";
  static DELIM_EQ = "=";
  static DELIM_VALUE = ",";
  static getThingFromToken = getThingFromToken;
  static getThingValues = getThingValues;
  static expandSubRegionDimThingList = expandSubRegionDimThingList;
  static getQueryStringFromParts = getQueryStringFromParts;
  static getMetadataKeyFromParts = getMetadataKeyFromParts;
  static normalizeMetadataKey = normalizeMetadataKey;

  constructor(
    entityClass,
    dimThingList,
    aggregate,
    queryStr,
    subRegionDimThingList = null,
    parentRegionConstraintList = null,
  ) {
    this.entityClass = entityClass;
    this.dimThingList = dimThingList;
    this.aggregate = aggregate;
    this.queryStr = queryStr;
    this.subRegionDimThingList = subRegionDimThingList;
    this.parentRegionConstraintList = parentRegionConstraintList;
  }

  toString() {
    return this.queryStr;
  }

  isSubsetOf(otherQuery) {
    return isQuerySubset(this, otherQuery);
  }

  static async fromString(queryString) {
    const parts = await parseQueryString(queryString);
    return new Query(
      parts.entityClass,
      parts.dimThingList,
      parts.aggregate,
      parts.queryString,
      parts.subRegionDimThingList,
      parts.parentRegionConstraintList,
    );
  }

  getMetadataKey() {
    return getMetadataKeyFromParts(
      this.entityClass,
      this.dimThingList,
      this.aggregate,
    );
  }

  getSubRegionFilter() {
    return getSubRegionFilter(this);
  }

  getParentRegionFilter() {
    return getParentRegionFilter(this);
  }

  static fromKeyValueList(keyValueList) {
    const parts = parseKeyValueList(keyValueList);
    return new Query(
      parts.entityClass,
      parts.dimThingList,
      parts.aggregate,
      parts.queryString,
    );
  }
}
