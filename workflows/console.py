from lanka_data import Db


def main():
    while True:
        cmd = input("> /")
        if cmd in ["x", "q"]:
            break

        try:
            Db(cmd)
        except Exception as e:
            print(f"‼️ Error: {e}")


if __name__ == "__main__":
    main()
