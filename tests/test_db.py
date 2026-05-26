import json
import os
import unittest

from lanka_data import Db


class TestCase(unittest.TestCase):
    def test_db(self):
        test_data = None
        with open(
            os.path.join("tests", "test_db.data.json"), "r", encoding="utf-8"
        ) as f:
            test_data = json.load(f)

        for cmd, expected_output_or_results in test_data.items():
            db = Db(cmd)
            actual_output = db.run(do_open_images=False, do_use_cache=False)
            if "results" in actual_output:
                print(json.dumps(actual_output, indent=2))
                actual_results = actual_output["results"]
                self.assertEqual(expected_output_or_results, actual_results)

            else:
                print(json.dumps(actual_output, indent=2))
                self.assertEqual(expected_output_or_results, actual_output)
