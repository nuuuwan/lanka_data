import hashlib
import os
import unittest

from lanka_data import Db
from utils_future import JSONFile

DATA_DIR = os.path.join("tests", "data")
CMDS_FILE = os.path.join("tests", "cmds.json")


def cmd_to_hash(cmd):
    return hashlib.md5(cmd.encode()).hexdigest()


def load_test_data():
    cmds = JSONFile(CMDS_FILE).read()
    entries = []
    for cmd in cmds:
        path = os.path.join(DATA_DIR, f"{cmd_to_hash(cmd)}.json")
        data = JSONFile(path).read()
        entries.append((data["cmd"], data["expected_output"]))
    return entries


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


for i, (cmd, expected_output) in enumerate(load_test_data()):
    safe_name = cmd.replace("/", "_").replace(":", "_")
    name = f"test_db_{i:03d}_{safe_name}"
    setattr(TestCase, name, make_test(cmd, expected_output))
