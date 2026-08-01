import Datum from "./Datum.js";
import ShallowDict from "../base/ShallowDict.js";

export default class DatumSet {
  constructor(datumList) {
    this.datumList = datumList;
  }

  static fromLankaData(lankaData) {
    const shallowDict = ShallowDict.fromDeep(lankaData);
    const datumList = shallowDict
      .entries()
      .map(([shallowKey, value]) => new Datum(shallowKey, value));
    return new DatumSet(datumList);
  }
}
