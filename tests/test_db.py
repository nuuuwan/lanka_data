import random
import unittest

from lanka_data import Db, Example


class TestCase(unittest.TestCase):
    pass


def make_test(cmd, expected_output):

    def test(self):
        db = Db(cmd)
        actual_output = db.run(do_open_images=False, do_use_cache=False)

        actual_output["query_time_ms"] = 0

        self.assertEqual(expected_output, actual_output)

        self.assertTrue(actual_output.get("result"))
        self.assertIsNone(actual_output.get("error"))
        if actual_output.get("result"):
            self.assertTrue(actual_output["result"]["source"])
            self.assertTrue(actual_output["result"]["source_url"])

    return test


output_idx = Example.get_output_idx()
MAX_TESTS_TO_RUN = 5
sampled_items = random.sample(
    list(output_idx.items()), min(MAX_TESTS_TO_RUN, len(output_idx))
)

for i, (cmd, expected_output) in enumerate(sampled_items):
    safe_name = cmd.replace("/", "_").replace(":", "_")
    name = f"test_db_{i:03d}_{safe_name}"
    setattr(TestCase, name, make_test(cmd, expected_output))


def _test_db_compare_years_json(self):
    output = Db("Religion/2024-2012/LK:district/JSON").run(
        do_open_images=False,
        do_use_cache=False,
    )
    self.assertIsNone(output.get("error"))

    result = output["result"]
    self.assertEqual("2024-2012", result["when_description"])

    aggr_data = result["aggr_data"]
    self.assertEqual({"2024", "2012"}, set(aggr_data["values_by_when"]))
    self.assertIn("Buddhist", aggr_data["delta_values"])
    self.assertIn("error", aggr_data)

    district_data = result["data_list"][0]
    self.assertEqual(
        district_data["error"],
        round(
            sum(district_data["error_values"].values())
            / len(district_data["error_values"]),
            8,
        ),
    )
    self.assertIn("pct_delta_values", district_data)


setattr(TestCase, "test_db_compare_years_json", _test_db_compare_years_json)
