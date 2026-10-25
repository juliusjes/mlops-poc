from copy import deepcopy
import json
import os
import sys

import requests
"""
Experiment steps:
    1. generate data with a data configuration

    2. start inference client
    3. set the desired drift function and data source via set_drift_func and set_data_source
    4. start inference server with defined detection algorithms and settings
    5. start orchestrator and run manual_start for each subset of data

    6. The stopping condition should be defined in the inference server such that the notifies will let the experiment runner know when to stop sending samples via the inference client

    7. Once all combinations are tested the trace logs should be parsed one big data source for analysis. This should be aggregated with the drift function to get gound truth sample indexing
    


"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Event, Thread
from time import sleep

import dotenv
import pandas as pd
from pydantic import BaseModel
import requests

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
dotenv.load_dotenv(REPOSITORY_ROOT / "ml-platform/.env")

from runtime import wait_for_terminal_event, write_state
"""
Experiment steps:
    0. start mlops-stack
    1. generate data with a data configuration

    2. start inference client
    3. set the desired drift function and data source via set_drift_func and set_data_source
    4. start inference server with defined detection algorithms and settings
    5. start orchestrator and run manual_start for each subset of data

    6. The stopping condition should be defined in the inference server such that the notifies will let the experiment runner know when to stop sending samples via the inference client

    7. Once all combinations are tested, the trace logs should be parsed one big data source for analysis. This should be aggregated with the drift function to get gound truth of the drift
    


- Preferrably the state of the experiment should be saved for the ability to resume if something happens.
    - The subsets should be correctly still in LakeFS
    - The model training should be deterministic
    - Return to the previous:
        - detection method
        - drift regime
        - subset
    - start from step 2

    - copying the trace logs after a succesul subset run to a target directory could be useful for this
"""


# variables
class Vars:
    #evidently_url: str  = os.environ["EVIDENTLY_ENDPOINT_URL"]
    mlflow_url: str     = os.environ["MLFLOW_TRACKING_URI"].replace("mlflow", "localhost")
    lakefs_url: str     = os.environ["SERVER_ENDPOINT_URL"].replace("lakefs", "localhost")
    inf_server_url: str = "http://localhost:8012"
    inf_client_url: str = "http://localhost:8013"
    orchestrator_url: str = "http://localhost:8014"

    temp_dir: Path = REPOSITORY_ROOT / "experiments/experiment2/temp/"
    log_dir: Path = REPOSITORY_ROOT / "outputs/logs/"
    data_config_file: Path = REPOSITORY_ROOT / "experiments/experiment2/data_config.json"

    #mlops_stack_start: Path = PROJECT_ROOT / "src/mlops-stack/start_all.sh"
    data_generation: Path = REPOSITORY_ROOT / "ml-platform/src/ml_platform/data_generation/generate_datasets.py"

    inference_client_settings: Path = "experiments/experiment2/inference_client_settings.json"
    inference_client_dir: Path = REPOSITORY_ROOT / "ml-platform/src/ml_platform/inference_client/"

    inference_server_settings: Path = "experiments/experiment2/inference_server_settings.json"
    manual_start: Path = REPOSITORY_ROOT / "experiments/experiment2/manual_start.json"
    retrain_config_file: Path = REPOSITORY_ROOT / "experiments/experiment2/retrain_cfg.json"


# step classes
class StepResult(BaseModel):
    success: bool
    results: dict = {}


class StepSuccess(StepResult):
    success: bool = True
    results: dict = {}


class StepFail(StepResult):
    success: bool = False
    results: dict = {}

    @classmethod
    def here(cls, **results):
        frame = sys._getframe(1)
        return cls( results={ "__file__": frame.f_code.co_filename, "__line__": frame.f_lineno, **results, } )


class StepFailedError(Exception):
    def __init__(self, step: StepFail):
        file = step.results.get("__file__", "<unknown>")
        line = step.results.get("__line__", "?")
        extra = { k: v for k, v in (step.results or {}).items() if not k.startswith("__") }
        msg = f"Step failed at {file}:{line}"
        if extra: msg += f"\nResults: {extra}"
        super().__init__(msg)
        self.step = step


stop_event = Event()

def create_server(check_fn, port=8016):
    stop_event.clear()

    class NotifyHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)

            try:
                data = json.loads(body)
                print("Received:", data)

                if check_fn(data):
                    stop_event.set()

            except Exception:
                print("Received non-JSON payload")

            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")

        def log_message(self, format, *args):
            pass

    return HTTPServer(("0.0.0.0", port), NotifyHandler)


# helpers
def check_service(url, code):
    try:
        response = requests.get(url)
    except Exception:
        return False
    return response.status_code == code


def get_event(event_id: str) -> dict:
    response = requests.get(
        f"{Vars.orchestrator_url}/events/{event_id}", timeout=30
    )
    response.raise_for_status()
    return response.json()


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


"""
Below are defined the steps described at the beginning of the file
"""

def step0(prev_result: StepResult=StepSuccess()) -> StepResult:
    """
    1. check if mlops-stack is active
    2. ping /health on all services
    """
    os.makedirs(Vars.temp_dir, exist_ok=True)

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
            return StepFail(results={"reason": f"{url} failed"})
        else:
            print(f"OK  : {url}: {ok}")

    return StepSuccess()


def step1(prev_result: StepResult) -> StepResult:
    """ Generate data"""
    if not prev_result.success:
        raise StepFailedError(prev_result)


    if not Vars.data_config_file.exists():
        print(Vars.data_config_file)
        return StepFail(results={"reason": "data_config not found"})
    
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

    if m is None:
        return StepFail()

    with open(Vars.data_config_file, "r") as f:
        cfg = json.load(f)

    final_stage = cfg["data_configuration"]["phases"][-1]
    stage = m.group("stage")

    if stage != final_stage: # make sure final stage mathces the configured final stage
        return StepFail()

    data_source = m.group("artifact").removeprefix("lakefs://")

    return StepSuccess( results={ "data_source": data_source, } )


# def step2(prev_result: StepResult):
#     """
#     Build and start inference client container, or use available instance
#     """
#     if not prev_result.success:
#         raise StepFailedError(prev_result)

#     if check_service(f"{Vars.inf_client_url}/health", 200):
#         return StepSuccess(results={**prev_result.results})

#     with podman.PodmanClient() as client:

#         # copy settings over
#         insert_line = 17
#         text = f"COPY {Vars.inference_client_settings} ./inference_client/settings.json"
#         cmd = [ "sed", "-i", f"{insert_line}i\\{text}", Vars.inference_client_dockerfile, ]
#         returncode, buffer = run_cmd_live(cmd)

#         if returncode != 0:
#             return StepFail( results={"reason": "Copying settings over failed"} )

#         cmd = ["sed", f"{insert_line}q;d", f"{Vars.inference_client_dockerfile}"]
#         returncode, buffer = run_cmd_live(cmd)

#         # build image
#         cmd = [ "bash", str(Vars.inference_client_dir / "build.sh"), ]
#         returncode, buffer = run_cmd_live(cmd)

#         cmd = [ "sed", "-i", f"{insert_line}d", Vars.inference_client_dockerfile, ]
#         sed_return, _ = run_cmd_live(cmd)

#         if sed_return != 0:
#             return StepFail(results={"reason": "deleting generated line in dockerfile failed"})

#         if returncode != 0:
#             return StepFail( results={"reason": "building inference client failed"} )

#         ic = client.containers.list(all=True, name="inference-client")

#         if len(ic) < 1:
#             return StepFail(results={"reason": "Inference-client image not available"})


    # # start inference client        
    # cmd = ["bash", str(Vars.inference_client_dir / "start_pod.sh")]
    # proc = run_cmd_live(cmd, wait=False, printout=False)

    # max_tries = 5
    # tries = 0
    # while not check_service(f"{Vars.inf_client_url}/health", 200):
    #     sleep(10)
    #     if tries >= max_tries:
    #         return StepFail(results={"reason": "inference server not available"})
    #     tries += 1
    
    # return StepSuccess(results={**prev_result.results})


# def step3(prev_result: StepResult) -> StepResult:
#     """
#     build and start inference server container, or use available instance
#     """
#     if not prev_result.success:
#         raise StepFailedError(prev_result)

#     if check_service(f"{Vars.inf_server_url}/state", 200):
#         return StepSuccess(results={**prev_result.results})

#     with podman.PodmanClient() as client:

#         # copy settings over
#         insert_line = 18
#         text = f"COPY {Vars.inference_server_settings} ./inference_server2/settings.json"
#         cmd = [ "sed", "-i", f"{insert_line}i\\{text}", Vars.inference_server_dockerfile, ]
#         returncode, buffer = run_cmd_live(cmd)

#         if returncode != 0:
#             return StepFail( results={"reason": "Copying settings over failed"} )

#         # build image
#         cmd = [ "bash", str(Vars.inference_server_dir / "build.sh"), ]
#         returncode, buffer = run_cmd_live(cmd)

#         cmd = [ "sed", "-i", f"{insert_line}d", Vars.inference_server_dockerfile, ]
#         sed_return, _ = run_cmd_live(cmd)

#         if sed_return != 0:
#             return StepFail(results={"reason": "deleting generated line in dockerfile failed"})

#         if returncode != 0:
#             return StepFail( results={"reason": "building inference sever failed"} )

#         ic = client.containers.list(all=True, name="inference-server2")

#         if len(ic) < 1:
#             return StepFail(results={"reason": "inference-server image not available"})


#     # start inference server        
#     cmd = ["bash", str(Vars.inference_server_dir / "start_pod.sh")]
#     run_cmd_live(cmd, wait=False, printout=False)

#     max_tries = 10
#     tries = 0
#     while not check_service(f"{Vars.inf_server_url}/state", 200):
#         print("checking inference server state")
#         if tries >= max_tries:
#             return StepFail(results={"reason": "inference server not available"})
#         sleep(10)
        
#         tries += 1

#     return StepSuccess(results={**prev_result.results})

    
# def step4(prev_result: StepResult):
#     """
#     Build and start orchestrator
#     """
#     if not prev_result.success:
#         raise StepFailedError(prev_result)
    
#     if check_service(f"{Vars.orchestrator_url}/health", 200):
#         print("Orchestrator OK")
#         return StepSuccess(results={**prev_result.results})


#     return StepSuccess(results={**prev_result.results})


def step5(prev_result: StepResult) -> StepResult:
    """
    Run experiment
    """
    if not prev_result.success:
        raise StepFailedError(prev_result)

    if not os.path.exists(Vars.manual_start):
        return StepFail(results={"reason": f"manual_start.json does not exits in {Vars.manual_start}"})
    
    with open(Vars.manual_start, "r") as f:
        manual_cfg = json.load(f)

    # set detection methods for inference server
    if not check_service(f"{Vars.inf_server_url}/state", 200):
        return StepFail(results={"reason": "inference server not available"})

    total_regimes =  len(EXPERIMENT_STATE["waiting"]["regime"])
    total_subsets = len(EXPERIMENT_STATE["waiting"]["subset"])
    total_iterations = total_subsets * total_regimes
    current_iteration = 1

    completed_pairs = set(
        zip(
            EXPERIMENT_STATE["completed"]["subset"],
            EXPERIMENT_STATE["completed"]["regime"],
        )
    )

    for i_subset, subset in enumerate(list(EXPERIMENT_STATE["waiting"]["subset"])):
        if all(
            (subset, regime) in completed_pairs
            for regime in EXPERIMENT_STATE["waiting"]["regime"]
        ):
            EXPERIMENT_STATE["waiting"]["subset"].remove(subset)
            write_state(Vars.temp_dir / "state.json", EXPERIMENT_STATE)
            print(f"Skipping completed subset: {subset}")
            continue

        # manual start to train the model and update to inference server

        off_data = prev_result.results["data_source"] + str(subset) + "/offline/"            
        manual_cfg["training_config"]["dataset_location"] = off_data # replace training data

        for v in manual_cfg["validation_config"]["validations"]: # replace validation data
            for r in v["rules"]:
                r["data_source"] = off_data
        
        r = requests.post(url=f"{Vars.orchestrator_url}/events/manual-start", json=manual_cfg)
        
        if r.status_code != 202:
            print(r.content)
            return StepFail(results={"reason": "manual-start failed"})
        
        event_id = r.json()["event_id"]

        print(f"Manual start, event_id={event_id}")

        try:
            event = wait_for_terminal_event(get_event, event_id)
        except (requests.RequestException, TimeoutError) as error:
            return StepFail(results={"reason": f"Training event unavailable: {error}"})

        if event["status"] == "failed":
            return StepFail(results={
                "reason": "Training the model failed",
                "event_id": event_id,
                "error": event.get("error"),
            })

        deployment_uuid = event.get("result", {}).get("uuid")
        if deployment_uuid is None:
            return StepFail(results={
                "reason": "Training completed without a deployment UUID",
                "event_id": event_id,
            })

        try:
            available_models = requests.get(
                f"{Vars.inf_server_url}/available_models", timeout=30
            ).json()["available"]
        except (requests.RequestException, KeyError, ValueError) as error:
            return StepFail(results={"reason": f"Inference server unavailable: {error}"})

        if deployment_uuid not in available_models:
            return StepFail(results={
                "reason": "Completed event deployment is not available in inference server",
                "event_id": event_id,
                "deployment_uuid": deployment_uuid,
            })

        print(f"Model trained and deployed ({available_models[deployment_uuid]})")

        cur_uuid = deployment_uuid

        data = {"uuid": cur_uuid, "configs": methods_settings}
        r = requests.post(url=f"{Vars.inf_server_url}/add_plugins", json=data)
        print(r.json())

        for i, s in enumerate(r.json()["success"]):
            if not s:
                return StepFail(results={"reason": f"failed to add detector {methods_settings[i]}"})
        

        for i_regime, drift_regime in enumerate(EXPERIMENT_STATE["waiting"]["regime"]):
            if (subset, drift_regime) in completed_pairs:
                print(
                    f"Skipping completed subset/regime combination: "
                    f"{subset}/{drift_regime}"
                )
                continue

            print(f"\nROUND {current_iteration}/{total_iterations}: subset {i_subset + 1}/{total_subsets}, regime {i_regime + 1}/{total_regimes}\n")

            # set online data and drift function for inference client
            on_data = prev_result.results["data_source"] + str(subset) + "/online/"

            r = requests.post(url=f"{Vars.inf_client_url}/set_drift_func", json=drifts[drift_regime])
            
            if not r.json()["found"]:
                return StepFail(results={"reason": f"failed to set drift function {drifts[drift_regime]}"})

            r = requests.post(url=f"{Vars.inf_client_url}/set_data_source",
                json={
                    "location": on_data,
                    "dataset_description": manual_cfg["training_config"]["dataset_description"]
                    }
                )
            if not r.json()["found"]:
                return StepFail(results={"reason": f"failed to set data source {on_data}"})


            detections = {
                "mmd": False,
            }

            def stopping_condition(data):
                if isinstance(data, str):
                    data = json.loads(data)
                if data["name"] in detections:
                    if not detections[data["name"]] and data["is_drift"]:
                        detections[data["name"]] = True 
                    
                return all(detections.values()) # wait until all detectors have fired

            stop_event.clear()
            print(f"stopping condition reset({stop_event.is_set()}):", detections)

            server = create_server(stopping_condition)

            Thread(target=server.serve_forever, daemon=True).start()

            print("Waiting for notification...")

            while not stop_event.is_set():
                
                response = requests.post(
                    f"{Vars.inf_client_url}/send_n_samples",
                    json={"n": 10, "uuid": cur_uuid, "labels": None},
                    timeout=30,
                )
                if response.status_code != 200:
                    print(response.content)
                    print("send_n_samples returned a non-success response")
                    break

                r = requests.get(f"{Vars.inf_client_url}/is_out_of_samples")
                if r.json()["out_of_samples"]:
                    csi = r.json()["current_sample_index"]
                    ts = r.json()["total_samples"]

                    r = requests.post(
                        url=f"{Vars.inf_server_url}/sync_plugins",
                        json={"uuid": cur_uuid},
                        timeout=30,
                    )

                    stop_event.set()

                sleep(0.1)

            print("Stopping condition achieved.")
            server.shutdown()
            del server

            # capture detection time
            r = requests.post(url=f"{Vars.inf_server_url}/reset_plugin_log", json={"uuid": cur_uuid})
            clog = r.json()["log_file"]

            r = requests.post(url=f"{Vars.inf_server_url}/reset_plugins", json={"uuid": cur_uuid})
            clog = Path(clog).name

            # add mmd back in reset state
            data = {"uuid": cur_uuid, "configs": methods_settings}
            r = requests.post(url=f"{Vars.inf_server_url}/add_plugins", json=data)
            print(r.json())


            EXPERIMENT_STATE["completed"]["associated_log_files"].append(clog)
            EXPERIMENT_STATE["completed"]["regime"].append(drift_regime)
            EXPERIMENT_STATE["completed"]["subset"].append(subset)
            EXPERIMENT_STATE["completed"]["model"].append(available_models[deployment_uuid])
            completed_pairs.add((subset, drift_regime))
            write_state(Vars.temp_dir / "state.json", EXPERIMENT_STATE)


            # generate pre and post detection datasets.
            data_directory = Vars.temp_dir
            os.makedirs(data_directory, exist_ok=True)
        
            dst = data_directory / f"{drift_regime}_{subset}_{clog}"
            src = Vars.log_dir / clog
            print("COPY", src, dst)
            shutil.copy(src, dst)
            df = pd.read_csv(dst)
            
            regex = r"^mmd.*\.(mmd2u|perc)$|^sample_index$"
            valid_df = df.dropna(subset=df.filter(regex=regex).columns)

            detection_index = None

            for c in valid_df.filter(regex=r"^mmd.*\.mmd2u$").columns:
                perc_col = c.rsplit(".", 1)[0] + ".perc"

                mask = valid_df[c] > valid_df[perc_col]

                if mask.any():
                    detection_index = int(valid_df.loc[mask, "sample_index"].iloc[0])
                    break

            if not detection_index:
                print("No detection was made, retraining eval skipped")
                continue

            print(f"Detection was made at index {detection_index}, evaluation starts")

            pre_loc = off_data.removesuffix("offline/") + f"pre_{drift_regime}/"
            print("Desired pre_loc", pre_loc)
            dsg_pre= {
                    "new_location": pre_loc,
                    "size": win_size,
                    "start": detection_index - win_size
                    }
            post_loc = off_data.removesuffix("offline/") + f"post_{drift_regime}/"
            print("Desired post_loc", post_loc)
            dsg_post = {
                    "new_location": post_loc,
                    "size": win_size,
                    "start": detection_index
                    }

            pre_r = requests.post(
                url=f"{Vars.inf_client_url}/generate_drifted_dataset",
                json=dsg_pre
                )

            post_r = requests.post(
                url=f"{Vars.inf_client_url}/generate_drifted_dataset",
                json=dsg_post
                )       

            pre_loc = pre_r.json()["new_location"]
            post_loc = post_r.json()["new_location"]

            # validate baseline model on pre and post datasets
            
            r = requests.get(url=f"{Vars.orchestrator_url}/events/{event_id}")
            mname, ver = r.json()["result"]["model_name"], r.json()["result"]["model_version"]
            baseline_uri = f"models:/{mname}/{ver}"


            vcfg = deepcopy(manual_cfg["validation_config"])
            vcfg["model_source"] = "mlflow"
            vcfg["model_load_kwargs"]["model_uri"] = baseline_uri

            # validate pre drift
            for validation in vcfg["validations"]:
                for rule in validation["rules"]:
                    rule["data_source"] = pre_loc
                    rule["n_samples"] = win_size

            print("Validating pre drift window")
            pre_base_event = requests.post(
                url=f"{Vars.orchestrator_url}/events/validate",
                json={"validation_config": vcfg}
                )
            
            # validate post drift
            for validation in vcfg["validations"]:
                for rule in validation["rules"]:
                    rule["data_source"] = post_loc
                    rule["n_samples"] = win_size

            print("Validating post drift window")
            post_base_event = requests.post(
                url=f"{Vars.orchestrator_url}/events/validate",
                json={"validation_config": vcfg}
                )

            # train retrained on pre and validate on post datasets
            retrain_tcfg = deepcopy(manual_cfg)

            retrain_tcfg["training_config"]["model_source_type"] = "mlflow"
            retrain_tcfg["training_config"]["model_load_kwargs"]["model_uri"] = baseline_uri
            retrain_tcfg["training_config"]["dataset_location"] = pre_loc
            retrain_tcfg["training_config"]["n_training_samples"] = 150
            retrain_tcfg["training_config"]["run_name"] = "Retraining_run"
            retrain_tcfg["training_config"]["epochs"] = 2
            retrain_tcfg["deploy_criteria"] = "no_deployment"
            retrain_tcfg["validation_config"]["model_source"] = "mlflow"
            retrain_tcfg["validation_config"]["model_load_kwargs"] = {"model_uri": baseline_uri}
            for validation in retrain_tcfg["validation_config"]["validations"]:
                for rule in validation["rules"]:
                    rule["data_source"] = post_loc

            with open(Vars.retrain_config_file, "w") as f:
                json.dump(retrain_tcfg, f)

            print("Retraining and validating")
            r_retrain = requests.post(
                url=f"{Vars.orchestrator_url}/events/manual-start",
                json=retrain_tcfg
            )
            retrain_event_id = r_retrain.json()["event_id"]
            try:
                event_results = {
                    event_id: get_event(event_id),
                    pre_base_event.json()["event_id"]: wait_for_terminal_event(
                        get_event, pre_base_event.json()["event_id"]
                    ),
                    post_base_event.json()["event_id"]: wait_for_terminal_event(
                        get_event, post_base_event.json()["event_id"]
                    ),
                    retrain_event_id: wait_for_terminal_event(get_event, retrain_event_id),
                }
            except (requests.RequestException, TimeoutError) as error:
                return StepFail(results={"reason": f"Validation event unavailable: {error}"})

            failed_event_ids = [
                identifier
                for identifier, event_result in event_results.items()
                if event_result["status"] == "failed"
            ]
            if failed_event_ids:
                return StepFail(results={
                    "reason": "Validation or retraining event failed",
                    "event_ids": failed_event_ids,
                })


            # extract validation results from db
            event_ids = [
                pre_base_event.json()["event_id"],
                post_base_event.json()["event_id"],
                retrain_event_id
            ]
            prefixes = ["prebase", "postbase", "postretr"]

            print("Gathering validation results")

            for pref, val_event_id in zip(prefixes, event_ids):
                val_res = event_results[val_event_id]["result"]["validation_results"]
                for i, (rk, rv) in enumerate(val_res.items()):
                    exp_key = rk.split("_", 1)[-1] + f"{drift_regime}_{subset}_{pref}"
                    exp_value = rv[1]

                    EXPERIMENT_STATE["completed"]["validation"][exp_key] = exp_value

                    print(exp_key, f"{exp_value:.2f}")

            write_state(Vars.temp_dir / "state.json", EXPERIMENT_STATE)

            print("Got results!")

            current_iteration += 1

        print(f"Removing deployment {cur_uuid}")
        r = requests.post(f"{Vars.inf_server_url}/remove_deployment", json={"uuid": cur_uuid})
        if r.json()["remaining"] != {}:
            return StepFail(results={"reason": "failed to remove deployment"})

        if all(
            (subset, regime) in completed_pairs
            for regime in EXPERIMENT_STATE["waiting"]["regime"]
        ):
            EXPERIMENT_STATE["waiting"]["subset"].remove(subset)
            write_state(Vars.temp_dir / "state.json", EXPERIMENT_STATE)

    return StepSuccess(results={
        **prev_result.results,
        "associated_log_files": EXPERIMENT_STATE["completed"]["associated_log_files"],
        "regimes": EXPERIMENT_STATE["completed"]["regime"],
        "subsets": EXPERIMENT_STATE["completed"]["subset"],
        "models": EXPERIMENT_STATE["completed"]["model"],
        "validations": EXPERIMENT_STATE["completed"]["validation"]
        })



if __name__ == "__main__":
    
    drifts = {
        "gradual": {
                "drift_func_name": "CPNDriftFunc",
                "drift_func_kwargs": {
                    "augmentation_func": ["draw_scratch", "blur"],
                    "augmentation_kwargs": [{
                        "n_scratches_range": [3,5],
                        "length_range": [0.3, 0.6],
                        "width_range": [0.01, 0.02],
                        "angle_range": [0, 6.2832],
                        "circularity_range": [0.0, 1.0],
                        "degree": 4,
                        "rng": "autoreplaced"
                    },
                    {}
                    ],
                    "parameter_names": ["curvature_range", "sigma"],
                    "param_values": [[0.0, 0.6], [0.0, 1.2]],
                    "param_cps": [[0.3, 0.7], [0.3, 0.7]],
                    "class_ratio": [0.5, 1.0],
                    "replace_all_labels": [True, False],
                }
            },
        "sudden": {
            "drift_func_name": "CPNDriftFunc",
            "drift_func_kwargs": {
                "augmentation_func": ["draw_scratch", "blur"],
                "augmentation_kwargs": [{
                    "n_scratches_range": [3,5],
                    "length_range": [0.3, 0.6],
                    "width_range": [0.01, 0.02],
                    "angle_range": [0, 6.2832],
                    "circularity_range": [0.0, 1.0],
                    "degree": 4,
                    "rng": "autoreplaced"
                },
                {}
                ],
                "parameter_names": ["curvature_range", "sigma"],
                "param_values": [[0.0, 0.6], [0.0, 1.2]],
                "param_cps": [[0.3, 0.31], [0.3, 0.31]],
                "class_ratio": [0.5, 1.0],
                "replace_all_labels": [True, False],
            }
        }
    }

    methods_settings = [
        {
            "plugin_type": "drift",
            "name": "mmd",
            "parameters": {
                "quantile_probability":0.95,
                "bootstrap": True
            },
            "trace_fields": ["mmd2u", "perc"],
            "notifiers": ["http://localhost:8016"],
        },
        {
            "plugin_type": "eval",
            "name": "metrics",
            "parameters": {
            },
            "trace_fields": [],
            "notifiers": [],
        }
    ]


    win_size= 300
    


    EXPERIMENT_STATE = {
        "completed": {
            "subset": [],
            "regime": [],
            "associated_log_files": [],
            "model": [],
            "combination": [],
            "validation": {},
            },

        "waiting": {
            "subset": [],
            "regime": [],
        }  
    }

    regimes = ["gradual", "sudden"]
    subsets = list(range(50))

    # return from previous state
    os.makedirs(Vars.temp_dir, exist_ok=True)
    state_file = Vars.temp_dir / "state.json"
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            EXPERIMENT_STATE = json.load(f)
        print(f"Loaded experiment state: {EXPERIMENT_STATE}")
    else:
        files = os.listdir(Vars.temp_dir)
        for f in files:
            full_path = Vars.temp_dir / f
            print(f"removing {full_path}")
            os.remove(full_path)

        EXPERIMENT_STATE["waiting"] = {
            "subset": subsets,
            "regime": regimes,
        }

    
    # RUN STEPS
    steps = [step0, step1, step5]

    previous = StepSuccess()

    for i, step in enumerate(steps):
        print(f"\nSTEP {i} PROCESSING\n")

        stepresult = step(previous)
        print(f"\nSTEP {i} DONE, result: {stepresult}\n")
        previous = stepresult

        

        # except KeyboardInterrupt:
        #     print(f"step{i} interrupted")
        #     print("experiment state:", EXPERIMENT_STATE)

        #     with open(state_file, "w") as f:
        #         json.dump(EXPERIMENT_STATE, f)
        #     break

        # except Exception as e:
        #     print(f"step{i} raised an exception: {e}")
        #     print("experiment state:", EXPERIMENT_STATE)

        #     with open(state_file, "w") as f:
        #         json.dump(EXPERIMENT_STATE, f)
        #     break

    write_state(state_file, EXPERIMENT_STATE)
            
