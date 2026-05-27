import os

from lanka_data import Db
from utils_future import JSONFile


def main():
    test_data_path = os.path.join("tests", "test_db.data.json")
    cmd_to_output = JSONFile(test_data_path).read()
    cmd_to_actual_output = {}
    for cmd in cmd_to_output:
        db = Db(cmd)
        actual_output = db.run(do_open_images=False, do_use_cache=False)
        cmd_to_actual_output[cmd] = actual_output

    test_data_temp_path = os.path.join("tests", "test_db.data.temp.json")
    JSONFile(test_data_temp_path).write(cmd_to_actual_output)


if __name__ == "__main__":
    main()
