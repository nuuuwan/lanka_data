import ThingFactoryContentsMixin from "./ThingFactoryContentsMixin.js";
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
    const delimIndex = keyValue.search(/[:=]/);
    if (delimIndex !== -1) {
      const className = keyValue.slice(0, delimIndex);
      const value = keyValue.slice(delimIndex + 1);
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

for (const [className, ThingClass] of Object.entries(
  ThingFactoryContentsMixin,
)) {
  ThingClass.className = className;
}
