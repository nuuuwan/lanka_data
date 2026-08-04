import { isQuerySubset } from "./QueryComparison.js";
import { getParentRegionFilter, getSubRegionFilter } from "./QueryFilters.js";
import { getMetadataKey, normalizeMetadataKey } from "./QueryMetadataKey.js";
import { parseQuery, queryFromKeyValues } from "./QueryParser.js";
import {
  expandSubRegionDimensions,
  getQueryString,
  getThingFromToken,
  getThingValues,
  QUERY_DELIMITERS,
} from "./QueryTokens.js";

export default class Query {
  static DELIM_TOKEN = QUERY_DELIMITERS.token;
  static DELIM_DIM = QUERY_DELIMITERS.dimension;
  static DELIM_EQ = QUERY_DELIMITERS.equal;
  static DELIM_VALUE = QUERY_DELIMITERS.value;

  constructor(
    entityClass,
    dimThingList,
    aggregate,
    queryStr,
    subRegionDimThingList = null,
    parentRegionConstraintList = null,
  ) {
    Object.assign(this, {
      entityClass,
      dimThingList,
      aggregate,
      queryStr,
      subRegionDimThingList,
      parentRegionConstraintList,
    });
  }

  toString() {
    return this.queryStr;
  }

  isSubsetOf(otherQuery) {
    return isQuerySubset(this, otherQuery);
  }

  static fromString(queryString) {
    return parseQuery(Query, queryString);
  }

  static getThingFromToken(token) {
    return getThingFromToken(token);
  }

  static getThingValues(thing) {
    return getThingValues(thing);
  }

  static expandSubRegionDimThingList(dimensions) {
    return expandSubRegionDimensions(dimensions);
  }

  static getQueryStringFromParts(entityClass, dimensions, aggregate) {
    return getQueryString(entityClass, dimensions, aggregate);
  }

  getMetadataKey() {
    return getMetadataKey(this.entityClass, this.dimThingList, this.aggregate);
  }

  static getMetadataKeyFromParts(entityClass, dimensions, aggregate) {
    return getMetadataKey(entityClass, dimensions, aggregate);
  }

  static normalizeMetadataKey(metadataKey) {
    return normalizeMetadataKey(metadataKey);
  }

  getSubRegionFilter() {
    return getSubRegionFilter(this);
  }

  getParentRegionFilter() {
    return getParentRegionFilter(this);
  }

  static fromKeyValueList(keyValues) {
    return queryFromKeyValues(Query, keyValues);
  }
}
