import json

from lanka_data import Db


def main():
    print("")
    print("/Where/What/When/How")
    print("")
    while True:
        cmd = input("> /")

        if cmd in ["x", "q"]:
            break

        if cmd in ["c"]:
            Db.cache_clear()
            continue

        output = Db(cmd).run(do_open_images=True, do_use_cache=True)
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
