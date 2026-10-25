import subprocess
from importlib import import_module

from ml_platform.data_generation.data_config import (
    DataConfiguration,
    PhaseConfiguration,
)
from ml_platform.shared.exceptions import PipelineException
from ml_platform.shared.paths import CONFIG_DIR, PLATFORM_ROOT
from ml_platform.shared.utils import info, load_object, make_hash
from pydantic import BaseModel

CWD = PLATFORM_ROOT
from ml_platform.shared import data_processing
from ml_platform.shared.services.lakefs_service import LakeFSService

# data_pro_module = importlib.import_module("data_processing")

def get_git_sha(short=False): # TODO: fails because direcotry not set
    try:
        cmd = ["git", "rev-parse"]
        if short: cmd.append("--short")
        cmd.append("HEAD")
        return subprocess.check_output( cmd, text=True, stderr=subprocess.DEVNULL, cwd=str(CWD) ).strip()
    except Exception:
        return None


def get_git_dirty(): # TODO: fails because direcotry not set
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
                cwd=str(CWD),
            ).strip()
        )
    except Exception:
        return False


def select_nested(d, paths):
    result = {}

    for path in paths:
        keys = path.split(".")

        current = d
        try:
            for key in keys:
                current = current[key]
        except (KeyError, TypeError):
            continue

        target = result
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = current

    return result


def replace_percent_inplace(obj, replacement):
    if isinstance(obj, BaseModel):
        for field in obj.model_fields:
            setattr(
                obj,
                field,
                replace_percent_inplace(getattr(obj, field), replacement),
            )

    elif isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = replace_percent_inplace(v, replacement)

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = replace_percent_inplace(v, replacement)

    elif isinstance(obj, str):
        return obj.replace("%", replacement)

    return obj




def initialize_data(dcfg: DataConfiguration):
    """
    Run the configred pipeline and save each phase to a new lakefs branch
    """
    lakefs_service = LakeFSService(verbose=True)

    # ensure repository
    repo = dcfg.repository
    if lakefs_service.get_repo(repo) is None:
        lakefs_service.create_repo(repo)
    
    phases = dcfg.phases

    prev_source_reference = "main"
    prev_uid = ""
    arg = None
    for phase in phases:
        info_prefix = f"[{phase}]"

        pcfg: PhaseConfiguration = dcfg.phase_configs.get(phase)

        if isinstance(pcfg.initial_arg, str):
            pcfg.initial_arg = pcfg.initial_arg.replace( "%", prev_uid)
        
        metadata_hashes={
            "git_hash": str(get_git_sha(short=True)),
            "git_dirty": str(get_git_dirty()),
            "pipe_hash": str(make_hash(pcfg.pipeline)),
            "pipe_args_hash": str(make_hash(pcfg.pipeline_kwargs)),
            "initial_arg_hash": str(make_hash(pcfg.initial_arg))
        } # TODO: maybe just use the whole pcfg for hash since that covers everything

        uid = make_hash(metadata_hashes)
        replace_percent_inplace(pcfg, uid)
        target_branch = pcfg.data_branch

        info(f"{phase} branch name is '{target_branch}'", prefix=info_prefix, tf=1)

        check = repo + "/" + target_branch

        if target_branch != "main/":# not lakefs_service.is_path_lakefs(check):
            info(f"Creating branch '{check}'", prefix=info_prefix, tf=1)
            branch = lakefs_service.create_branch(repo, target_branch, prev_source_reference)
        else:
            branch = lakefs_service.get_branch(repo, target_branch)

        target = pcfg.final_arg

        cond1 = not lakefs_service.is_path_lakefs(target)

        condition = (cond1)

        if not condition: # TODO: add forced duplication
            arg = pcfg.initial_arg if arg is None else arg
            for name in pcfg.pipeline: # run through pipeline
                try:
                    mod, nam = name.split(".")
                    module = getattr(data_processing, mod)
                    func = getattr(module, nam)

                    kwargs = pcfg.pipeline_kwargs.get(name, {})

                    if "metadata" in kwargs.keys(): # workaround to gather selected metadata
                        kwargs["metadata"] = select_nested(pcfg, pcfg.metadata_select)

                    message = f"Passing to '{name}', with {type(arg)} argument and {len(kwargs.keys())} kwargs"
                    info(message, prefix=info_prefix, tf=1)
                    print("KWARGS:", kwargs)
                    arg = func(arg, **kwargs)
                except Exception as e:
                    raise PipelineException(f"{e}, func: {name}, kwargs: {kwargs}")
                    # TODO: add revert here if the pipe fails

            msg = f"Autocommit of {phase} data by {__file__}"
            lakefs_service.make_commit(branch, msg, metadata_hashes) # Assumes that each phase alters the branch

        else:
            info(f"{phase} data in place already, skipping", prefix=info_prefix, tf=1)

        prev_uid = uid
        prev_source_reference = target_branch.rstrip("/")

        info_arg = pcfg.final_arg if arg is None else arg
        info(f"{phase} stage done, artifact in {info_arg}", prefix=info_prefix, tf=1)
        assert lakefs_service.is_path_lakefs(target) is True, f"{phase} result not matching" # only chechks for path


# TODO:
# Revisit experiment dataset generation after running a meaningful number of
# experiments.
#
# Current design:
# - One pipeline stage generates N replicate datasets (different seeds).
# - All replicates are stored in a single lakeFS branch.
# - The seed list contributes to the experiment UID.
#
# Potential issues:
# - Requesting a subset/superset of an existing seed set creates a new branch,
#   even when some or all datasets already exist.
# - The experiment-generation stage produces multiple artifacts rather than a
#   single dataset.
#
# Questions to revisit:
# - Should seeds be treated as part of the experiment identity or as individual
#   realizations/runs within a single experiment?
# - Should experiment branches be keyed only by configuration, with seeds stored
#   as sub-artifacts (allowing reuse of existing seeds)?
# - Is generating multiple datasets in one pipeline stage causing maintenance,
#   reproducibility, or workflow issues in practice?
#
# Defer changes until actual experiments reveal whether these are real
# limitations rather than theoretical concerns.

"""
0. check that LakeFS does not hold the raw data already
    - create the target repository
    - download raw data, convert and write the parquet to LakeFS directly
        - image conversion via numpy array -> bytes
    -> raw data branch
1. Perform filtering for size and padding for aspect ratio (check for existing copies)
    - access the data from LakeFS directly
    - create a new branch for filtered data
    -> filtered data branch
2. Create experiment splits
    - For each split seed in experiment config create a new branch
    - using filtered data directly from LakeFS, split the data to online and offline datasets
    - simulate scratches and upscale images to uniform size
    - write to the corresponding branch 
    -> N experiment branches
3. Run the ML experiment for each branch
    - train base model with offline data, store in model registry
        - data deserialisation similarly as the serialization
    - deploy model if passes base checks
    - initialize inference client with offline data directly from LakeFS parquet
    - start sending samples to inference service and assess drift signal
    - update offline data with some delay
    - retrain the model if drift is detected
        - if the stopping criterion is last or first detection, stop sending samples and conclude
    - continue with retrained model
        - if multiple detectors were selected this requires branching
"""

if __name__ == "__main__":
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser()
    #parser.add_argument("--env", default=ENV_FILE, help="Path to .env file")
    parser.add_argument("--config", default=CONFIG_DIR / "test_data_config.json", help="Path to config JSON")

    args = parser.parse_args()
    #assert os.path.isfile(args.env), f"{args.env} is not a file"
    assert os.path.isfile(args.config), f"{args.config} is not a file"
    
    #dotenv.load_dotenv(Path(args.env))
    with open(args.config, "r") as f: cfg = json.load(f)
    dcfg = DataConfiguration.model_validate(cfg["data_configuration"])
    
    # TODO: validate pipe config
    initialize_data(dcfg)
