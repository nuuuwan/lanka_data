import json
import time

from lanka_data import Db


def main():
    print("")
    print("/Where/What/When/How")
    print("")
    while True:
        cmd = input("> /")

        t_start = time.time()
        if cmd in ["x", "q"]:
            break

        try:
            result = Db(cmd).run()
            query_time_ms = int((time.time() - t_start) * 1000)
            print(
                json.dumps(
                    dict(
                        result=result,
                        query_time_ms=query_time_ms,
                    ),
                    indent=2,
                )
            )
        except Exception as e:
            query_time_ms = int((time.time() - t_start) * 1000)
            print(
                json.dumps(
                    dict(
                        error=str(e),
                        query_time_ms=query_time_ms,
                    ),
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
