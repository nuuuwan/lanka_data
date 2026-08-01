import Datum from "./Datum.js";
import ShallowDict from "../base/ShallowDict.js";

export default class DatumSet {
  constructor(datumList) {
    this.datumList = datumList;
  }

  static fromLankaData(lankaData) {
    const shallowDict = ShallowDict.fromDeep(lankaData);
    const datumList = Array.from(shallowDict.entries()).map((entry) =>
      Datum.fromShallowDictEntry(entry),
    );
    return new DatumSet(datumList);
  }
}
