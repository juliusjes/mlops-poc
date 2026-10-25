import pandas as pd
import pytest
import numpy as np
import re
import subprocess
from ml_platform.shared.paths import PROJECT_ROOT
from ml_platform.shared.services.lakefs_service import LakeFSService


@pytest.fixture
def lkfs():
    return LakeFSService()


def run_cmd_live(cmd, printout=True, wait=True):
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if not wait:
        return proc

    buffer = []
    for line in proc.stdout:
        if printout:
            print(line, end="")
        buffer.append(line)

    returncode = proc.wait()
    return returncode, buffer


def INACTIVE_test_e2e_data_integrity(lkfs):
    dg_paht = PROJECT_ROOT / "ml-platform/src/ml_platform/data_generation/generate_datasets.py"
    dcfg_path = PROJECT_ROOT / "ml-platform/tests/shared/LWMD_data_config.json"
    cmd = ["python3", dg_paht, "--config", dcfg_path]

    rc, buf = run_cmd_live(cmd)
    assert rc == 0

    i, m = 1, None
    pattern = (
            r"\[INFO\] "
            r"\[\d{2}:\d{2}:\d{2}\] "
            r"\[(?P<stage>[^\]]+)\]: "
            r"(?P=stage) stage done, artifact in (?P<artifact>\S+)"
        )
    while m is None and i < 10:
        final_output = buf[-i]
        print(final_output)
        m = re.search(pattern, final_output) # NOTE: reliant on printed output
        i += 1

    assert m is not None

    final_stage = "experiment"
    stage = m.group("stage")

    #print(stage, final_stage)
    assert stage == final_stage

    data_source = m.group("artifact").removeprefix("lakefs://")

    for seed in [0, 1]:
        print(f"Seed {seed}")
        loc_off = data_source + f"{seed}/offline/"
        loc_on = data_source + f"{seed}/online/"

        df_off = lkfs.lakeFS_parquet_to_df(loc_off, deser_col=["waferMap"])
        df_on = lkfs.lakeFS_parquet_to_df(loc_on, deser_col=["waferMap"])

        off_counts = df_off["failure_label"].value_counts(dropna=False)
        # Failure-label distributions
        print("\tOFF failure_label counts:")
        print(f"\t{off_counts}")

        on_counts = df_on["failure_label"].value_counts(dropna=False)
        print("\tON failure_label counts:")
        print(f"\t{on_counts}")

        # Find waferMaps that occur in both dataframes
        off_maps = {x.tobytes() for x in df_off["waferMap"]}
        on_maps = {x.tobytes() for x in df_on["waferMap"]}

        matching = off_maps & on_maps
        print(f"\t Uniques in OFF: {len(off_maps)}")
        print(f"\t Uniques in ON: {len(on_maps)}")
        print(f"\tNumber of matching waferMaps: {len(matching)}")
        print(f"\tAny matches: {bool(matching)}")


def test_simple_data_integrity(lkfs):
    dg_paht = PROJECT_ROOT / "ml-platform/src/ml_platform/data_generation/generate_datasets.py"
    dcfg_path = PROJECT_ROOT / "ml-platform/tests/shared/simple_data_config.json"
    cmd = ["python3", dg_paht, "--config", dcfg_path]

    rc, buf = run_cmd_live(cmd)
    assert rc == 0

    i, m = 1, None
    pattern = (
            r"\[INFO\] "
            r"\[\d{2}:\d{2}:\d{2}\] "
            r"\[(?P<stage>[^\]]+)\]: "
            r"(?P=stage) stage done, artifact in (?P<artifact>\S+)"
        )
    while m is None and i < 10:
        final_output = buf[-i]
        print(final_output)
        m = re.search(pattern, final_output) # NOTE: reliant on printed output
        i += 1

    assert m is not None

    final_stage = "exp"
    stage = m.group("stage")

    #print(stage, final_stage)
    assert stage == final_stage

    data_source = m.group("artifact").removeprefix("lakefs://")

    for seed in [0, 1]:
        print(f"Seed {seed}")
        loc_off = data_source + f"{seed}/offline/"
        loc_on = data_source + f"{seed}/online/"

        df_off = lkfs.lakeFS_parquet_to_df(loc_off, deser_col=["X"])
        df_on = lkfs.lakeFS_parquet_to_df(loc_on, deser_col=["X"])

        off_counts = df_off["y"].value_counts(dropna=False)
        # Failure-label distributions
        print("\tOFF y counts:")
        print(f"\t{off_counts}")

        on_counts = df_on["y"].value_counts(dropna=False)
        print("\tON y counts:")
        print(f"\t{on_counts}")

        # Find waferMaps that occur in both dataframes
        off_maps = {x.tobytes() for x in df_off["X"]}
        on_maps = {x.tobytes() for x in df_on["X"]}

        matching = off_maps & on_maps
        print(f"\t Uniques in OFF: {len(off_maps)}")
        print(f"\t Uniques in ON: {len(on_maps)}")
        print(f"\tNumber of matching X: {len(matching)}")
        print(f"\tAny matches: {bool(matching)}")


    
if __name__ == "__main__":
    test_simple_data_integrity(LakeFSService())
    #test_e2e_data_integrity(LakeFSService())