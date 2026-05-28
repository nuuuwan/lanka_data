import os
import unittest

from lanka_data import Db
from utils_future import JSONFile

DATA_DIR = os.path.join("tests", "data")


def load_test_data():
    cmd_to_output = {}
    for f in sorted(os.listdir(DATA_DIR)):
        if f.endswith(".json"):
            path = os.path.join(DATA_DIR, f)
            cmd = f[:-5].replace(".", "/")  # strip .json, restore /
            cmd_to_output[cmd] = JSONFile(path).read()
    return cmd_to_output


class TestCase(unittest.TestCase):
    pass


def make_test(cmd, expected_output):
    def test(self):
        db = Db(cmd)
        actual_output = db.run(do_open_images=False, do_use_cache=False)

        expected_output["query_time_ms"] = None
        actual_output["query_time_ms"] = None

        self.assertEqual(expected_output, actual_output)

        self.assertTrue(
            actual_output.get("result") or actual_output.get("error")
        )
        if actual_output.get("result"):
            self.assertTrue(actual_output["result"]["source"])
            self.assertTrue(actual_output["result"]["source_url"])

    return test


for i, (cmd, expected_output) in enumerate(load_test_data().items()):
    safe_name = cmd.replace("/", "_").replace(":", "_")
    name = f"test_db_{i:03d}_{safe_name}"
    setattr(TestCase, name, make_test(cmd, expected_output))
