import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--test", required=True)

args = parser.parse_args()

print(f"Test value: {args.test}")