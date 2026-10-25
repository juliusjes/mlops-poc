import argparse
import logging

from ml_platform.workers.validator.engine import Validator
parser = argparse.ArgumentParser()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s %(asctime)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    # handlers=[
    #     logging.FileHandler(filename),
    #     logging.StreamHandler(sys.stdout), 
    # ],
    force=True
)

parser.add_argument("--jobid", required=True)

args = parser.parse_args()

Validator(args.jobid)
