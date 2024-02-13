import sys
from engineer.utils import rglob, load_module
import os


def main():
    if len(sys.argv) != 2:
        raise ValueError("Please provide path to run files.")
    path = sys.argv[1]
    if os.path.isdir(path):
        files = tuple(rglob(path, "*.py"))
    else:
        files = [path]
    print(f"Running tests in {len(files)} files.")
    for f in files:
        module = f[:-3].replace("/", ".")
        test = load_module(module + ".test")
        test()

    print("All tests completed!")


if __name__ == "__main__":
    main()
