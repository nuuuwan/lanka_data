require("@babel/register")({
  extensions: [".js"],
  ignore: [/node_modules/],
  presets: [require.resolve("babel-preset-react-app")],
});

const {
  createRawJSONMiddleware,
} = require("./nonview/core/raw_json/RawJSONMiddleware.js");

module.exports = function setupProxy(app) {
  app.use(createRawJSONMiddleware());
};
