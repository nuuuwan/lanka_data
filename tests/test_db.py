import os
import unittest

from lanka_data import Db
from utils_future import JSONFile


def load_test_data():
    return JSONFile(os.path.join("tests", "test_db.data.json")).read()


class TestCase(unittest.TestCase):
    pass


def make_test(cmd, expected_output):
    def test(self):
        db = Db(cmd)
        actual_output = db.run(do_open_images=False, do_use_cache=False)

        expected_output["query_time_ms"] = None
        actual_output["query_time_ms"] = None

        self.assertEqual(expected_output, actual_output)

    return test


for i, (cmd, expected_output) in enumerate(load_test_data().items()):
    name = f"test_db_{i:03d}_{cmd}"
    setattr(TestCase, name, make_test(cmd, expected_output))
