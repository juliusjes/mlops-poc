import argparse
import logging

from ml_platform.workers.trainer.engine import Trainer

parser = argparse.ArgumentParser()
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    # handlers=[
    #     logging.FileHandler(filename),
    #     logging.StreamHandler(sys.stdout), 
    # ],
    force=True
)
parser.add_argument("--jobid", required=True)
parser.add_argument("--postgresurloverride", required=False)
args = parser.parse_args()

Trainer(
    job_id=args.jobid,
    postgres_override_url=args.postgresurloverride
    )
