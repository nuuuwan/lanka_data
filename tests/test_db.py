import os
import unittest

from lanka_data import Db
from utils_future import JSONFile


def load_test_data():
    return JSONFile(os.path.join("tests", "test_db.data.json")).read()


class TestCase(unittest.TestCase):
    pass


def make_test(cmd, expected):
    def test(self):
        db = Db(cmd)
        actual_output = db.run(do_open_images=False, do_use_cache=False)
        if "results" in actual_output:
            self.assertEqual(expected, actual_output["results"])
        else:
            self.assertEqual(expected, actual_output)

    return test


for i, (cmd, expected) in enumerate(load_test_data().items()):
    name = f"test_db_{i:03d}_{cmd}"
    setattr(TestCase, name, make_test(cmd, expected))
