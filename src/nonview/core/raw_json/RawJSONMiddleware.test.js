import {
  createRawJSONMiddleware,
  getQueryStringFromRawJSONURL,
} from "./RawJSONMiddleware.js";

const RAW_URL =
  "/lanka_data/Vote/ElectionType+Time=1994+ED+Party/Count/JSON/raw.json";
const QUERY_STRING = "Vote/ElectionType+Time=1994+ED+Party/Count";

test("extracts the query from a raw JSON URL", () => {
  expect(getQueryStringFromRawJSONURL(RAW_URL)).toBe(QUERY_STRING);
  expect(
    getQueryStringFromRawJSONURL("/lanka_data/Vote/Count/JSON"),
  ).toBeNull();
});

test("returns queried data as a JSON attachment", async () => {
  const datumSet = { datumList: [{ count: 42 }] };
  const query = { id: "query" };
  const ensureRegionData = jest.fn().mockResolvedValue();
  const parseQuery = jest.fn().mockResolvedValue(query);
  const getDatumSet = jest.fn().mockResolvedValue(datumSet);
  const next = jest.fn();
  const response = {
    json: jest.fn(),
    send: jest.fn(),
    set: jest.fn(),
    status: jest.fn(),
  };
  const middleware = createRawJSONMiddleware({
    ensureRegionData,
    getDatumSet,
    parseQuery,
  });

  await middleware({ originalUrl: RAW_URL }, response, next);

  expect(next).not.toHaveBeenCalled();
  expect(ensureRegionData).toHaveBeenCalledTimes(1);
  expect(parseQuery).toHaveBeenCalledWith(QUERY_STRING);
  expect(getDatumSet).toHaveBeenCalledWith(query);
  expect(response.status).toHaveBeenCalledWith(200);
  expect(response.set).toHaveBeenCalledWith(
    "Content-Type",
    "application/json; charset=utf-8",
  );
  expect(response.set).toHaveBeenCalledWith(
    "Content-Disposition",
    'attachment; filename="raw.json"',
  );
  expect(response.send).toHaveBeenCalledWith(JSON.stringify(datumSet, null, 2));
});

test("passes unrelated URLs to the next middleware", async () => {
  const next = jest.fn();
  const middleware = createRawJSONMiddleware();

  await middleware({ url: "/lanka_data/" }, {}, next);

  expect(next).toHaveBeenCalledTimes(1);
});
