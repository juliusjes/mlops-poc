import json
import os
import re
import subprocess
import sys
from pathlib import Path

import dotenv
import podman
import requests
from ml_platform.shared.paths import ENV_FILE, PLATFORM_ROOT

dotenv.load_dotenv(ENV_FILE)

class Vars:
    container_names: tuple = [
        "ml-platform-postgres-1",
        "ml-platform-mlflow-1",
        "ml-platform-seaweed-1",
        "ml-platform-lakefs-1",
        "ml-platform-orchestrator-1",
        "ml-platform-inference-server-1"
    ]

    #evidently_url: str  = os.environ["EVIDENTLY_ENDPOINT_URL"]
    mlflow_url: str     = os.environ["MLFLOW_TRACKING_URI"].replace("mlflow", "localhost")
    lakefs_url: str     = os.environ["SERVER_ENDPOINT_URL"].replace("lakefs", "localhost")
    inf_server_url: str = "http://localhost:8012"
    inf_client_url: str = "http://localhost:8013"
    orchestrator_url: str = "http://localhost:8014"

    data_config_file: Path = PLATFORM_ROOT / "tests/end2end/test_data_config.json"
    data_generation: Path = PLATFORM_ROOT / "src/ml_platform/data_generation/generate_datasets.py"

def check_service(url, code):
    try:
        response = requests.get(url)
    except Exception as e:
        return False
    return response.status_code == code


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


def check_containers():
    with podman.PodmanClient() as client:
        if client.ping():
            container_names = set([c.name for c in client.containers.list()])
            required = set(Vars.container_names)
            if not container_names.issuperset(required):
                raise ValueError(f"required contrainers not running\nrunning:\n{container_names} \n, required:\n {required}")
        else:
            raise ValueError("Podman not available")

    urls = [
        Vars.lakefs_url + "/health",
        Vars.inf_server_url + "/state",
        Vars.inf_client_url + "/health",
        Vars.mlflow_url + "/health",
        Vars.orchestrator_url + "/health"
    ]

    for url in urls:
        ok = check_service(url, 200)
        if not ok:
            print(f"FAIL: {url}: {ok}")
            raise ValueError(f"{url} was not healthy")
        else:
            print(f"OK  : {url}: {ok}")


def generate_simple_data():
    """ Generate data"""
    assert Vars.data_config_file.exists(), f"{Vars.data_config_file} does not exist"
    
    cmd = [ sys.executable, Vars.data_generation, "--config", Vars.data_config_file, ]
    returncode, buffer = run_cmd_live(cmd)

    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)

    i, m = 1, None
    pattern = (
            r"\[INFO\] "
            r"\[\d{2}:\d{2}:\d{2}\] "
            r"\[(?P<stage>[^\]]+)\]: "
            r"(?P=stage) stage done, artifact in (?P<artifact>\S+)"
        )
    while m is None and i < 10:
        final_output = buffer[-i]
        print(final_output)
        m = re.search(pattern, final_output) # NOTE: reliant on printed output
        i += 1

    assert m is not None, f"Couldn't find matching final output string"

    with open(Vars.data_config_file, "r") as f:
        cfg = json.load(f)

    final_stage = cfg["data_configuration"]["phases"][-1]
    stage = m.group("stage")

    assert stage != final_stage, f"Final data generation stage doesn't match {stage}"
       
    data_source = m.group("artifact").removeprefix("lakefs://")

    return data_source
