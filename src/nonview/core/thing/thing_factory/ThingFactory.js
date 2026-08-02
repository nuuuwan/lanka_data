import ThingFactoryContentsMixin from "./ThingFactoryContentsMixin.js";
import KeyValue from "../../KeyValue.js";
import Thing from "../Thing.js";

export default class ThingFactory {
  static fromKey(key) {
    const ThingClass = ThingFactory[key];
    if (!ThingClass) {
      throw new Error(`ThingClass "${key}" not found in ThingFactory`);
    }
    return ThingClass;
  }

  static fromKeyValue(keyValue) {
    if (keyValue.includes(KeyValue.DELIM)) {
      const [className, value] = keyValue.split(KeyValue.DELIM);
      const ThingClass = ThingFactory[className];
      if (!ThingClass) {
        throw new Error(`ThingClass "${className}" not found in ThingFactory`);
      }
      return ThingClass.fromValue(value);
    }

    const ThingClass = ThingFactory.fromKey(keyValue);
    return ThingClass.fromValue(Thing.WILDCARD);
  }
}

Object.assign(ThingFactory, ThingFactoryContentsMixin);
