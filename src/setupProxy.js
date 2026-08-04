const { createJiti } = require("jiti");
const jiti = createJiti(__filename);
const { createRawJSONMiddleware } = jiti(
  "./nonview/core/raw_json/RawJSONMiddleware.js",
);

module.exports = function setupProxy(app) {
  app.use(createRawJSONMiddleware());
};
