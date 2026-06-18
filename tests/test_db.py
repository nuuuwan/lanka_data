import unittest

from lanka_data import Command, Example


class TestCase(unittest.TestCase):
    pass


def make_test(cmd, expected_output):

    def test(self):
        db = Command.from_str(cmd)
        actual_output = db.run(do_open_images=False, do_use_cache=False)

        actual_output["query_time_ms"] = 0
        actual_output["cache_hit"] = False
        expected_output["query_time_ms"] = 0
        expected_output["cache_hit"] = False

        self.assertEqual(expected_output, actual_output)

        self.assertTrue(actual_output.get("result"))
        self.assertIsNone(actual_output.get("error"))

    return test


output_idx = Example.get_cmd_to_output()

for i, (cmd, expected_output) in enumerate(output_idx.items(), start=1):
    safe_name = cmd.replace("/", "_").replace(":", "_")
    name = f"test_db_{i:03d}_{safe_name}"
    setattr(TestCase, name, make_test(cmd, expected_output))
