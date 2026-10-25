import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import re
from time import sleep
import time
import dotenv
import podman
from pydantic import BaseModel
import requests
import inspect
from paths import ENV_FILE, LOG_DIR, PROJECT_ROOT
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Event, Thread
import json
import shutil
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
dotenv.load_dotenv(ENV_FILE)

class Vars:
    container_names: tuple = [
        "postgresdb",
        "mlflow",
        "seaweed",
        "evidently",
        "lakefs"
    ]

    evidently_url: str  = os.environ["EVIDENTLY_ENDPOINT_URL"]
    mlflow_url: str     = os.environ["MLFLOW_TRACKING_URI"]
    lakefs_url: str     = os.environ["SERVER_ENDPOINT_URL"]
    inf_server_url: str = os.environ["INFERENCE_SERVER_URL"]
    inf_client_url: str = os.environ["INFERENCE_CLIENT_URL"]
    orchestrator_url: str = "http://localhost:8014"

    temp_dir: Path = PROJECT_ROOT / "experiments/experiment0/temp/"

    data_config_file: Path = PROJECT_ROOT / "experiments/experiment0/data_config.json"

    mlops_stack_start: Path = PROJECT_ROOT / "src/mlops-stack/start_all.sh"
    data_generation: Path = PROJECT_ROOT / "src/data_generation/generate_datasets.py"

    inference_client_settings: Path = "experiments/experiment0/inference_client_settings.json"
    inference_client_dir: Path = PROJECT_ROOT / "src/inference_client/"
    inference_client_dockerfile: Path = PROJECT_ROOT / "src/Dockerfile-inference-client.Dockerfile"

    inference_server_settings: Path = "experiments/experiment0/inference_server_settings.json"
    inference_server_dir: Path = PROJECT_ROOT / "src/inference_serverj2/"
    inference_server_dockerfile: Path = PROJECT_ROOT / "src/Dockerfile-inference-server2.Dockerfile"

    manual_start: Path = PROJECT_ROOT / "experiments/experiment0/manual_start.json"


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

    with podman.PodmanClient() as client:
        if client.ping():
            container_names = set([c.name for c in client.containers.list()])
            required = set(Vars.container_names)
            if not container_names.issuperset(required):
                print(f"{container_names} running, required {required}")
                return StepFail()
        else:
            return StepFail()

    urls = [
        Vars.lakefs_url + "/health",
        # Vars.inf_server_url + "/state",
        # Vars.inf_client_url + "/health",
        Vars.mlflow_url + "/health",
        Vars.evidently_url + "",
    ]

    for url in urls:
        ok = check_service(url, 200)
        if not ok:
            print(f"FAIL: {url}: {ok}")
            return StepFail()
        else:
            print(f"OK  : {url}: {ok}")

    return StepSuccess()


def step1(prev_result: StepResult) -> StepResult:
    if not prev_result.success:
        raise StepFailedError(prev_result)


    if not Vars.data_config_file.exists():
        return StepFail()
    
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


def step2(prev_result: StepResult):
    """
    Build and start inference client container, or use available instance
    """
    if not prev_result.success:
        raise StepFailedError(prev_result)

    if check_service(f"{Vars.inf_client_url}/health", 200):
        return StepSuccess(results={**prev_result.results})

    with podman.PodmanClient() as client:

        # copy settings over
        insert_line = 17
        text = f"COPY {Vars.inference_client_settings} ./inference_client/settings.json"
        cmd = [ "sed", "-i", f"{insert_line}i\\{text}", Vars.inference_client_dockerfile, ]
        returncode, buffer = run_cmd_live(cmd)

        if returncode != 0:
            return StepFail( results={"reason": "Copying settings over failed"} )

        cmd = ["sed", f"{insert_line}q;d", f"{Vars.inference_client_dockerfile}"]
        returncode, buffer = run_cmd_live(cmd)

        # build image
        cmd = [ "bash", str(Vars.inference_client_dir / "build.sh"), ]
        returncode, buffer = run_cmd_live(cmd)

        cmd = [ "sed", "-i", f"{insert_line}d", Vars.inference_client_dockerfile, ]
        sed_return, _ = run_cmd_live(cmd)

        if sed_return != 0:
            return StepFail(results={"reason": "deleting generated line in dockerfile failed"})

        if returncode != 0:
            return StepFail( results={"reason": "building inference client failed"} )

        ic = client.containers.list(all=True, name="inference-client")

        if len(ic) < 1:
            return StepFail(results={"reason": "Inference-client image not available"})


    # start inference client        
    cmd = ["bash", str(Vars.inference_client_dir / "start_pod.sh")]
    proc = run_cmd_live(cmd, wait=False, printout=False)

    max_tries = 5
    tries = 0
    while not check_service(f"{Vars.inf_client_url}/health", 200):
        sleep(10)
        if tries >= max_tries:
            return StepFail(results={"reason": "inference server not available"})
        tries += 1
    
    return StepSuccess(results={**prev_result.results})


def step3(prev_result: StepResult) -> StepResult:
    """
    build and start inference server container, or use available instance
    """
    if not prev_result.success:
        raise StepFailedError(prev_result)

    if check_service(f"{Vars.inf_server_url}/state", 200):
        return StepSuccess(results={**prev_result.results})

    with podman.PodmanClient() as client:

        # copy settings over
        insert_line = 18
        text = f"COPY {Vars.inference_server_settings} ./inference_server2/settings.json"
        cmd = [ "sed", "-i", f"{insert_line}i\\{text}", Vars.inference_server_dockerfile, ]
        returncode, buffer = run_cmd_live(cmd)

        if returncode != 0:
            return StepFail( results={"reason": "Copying settings over failed"} )

        # build image
        cmd = [ "bash", str(Vars.inference_server_dir / "build.sh"), ]
        returncode, buffer = run_cmd_live(cmd)

        cmd = [ "sed", "-i", f"{insert_line}d", Vars.inference_server_dockerfile, ]
        sed_return, _ = run_cmd_live(cmd)

        if sed_return != 0:
            return StepFail(results={"reason": "deleting generated line in dockerfile failed"})

        if returncode != 0:
            return StepFail( results={"reason": "building inference sever failed"} )

        ic = client.containers.list(all=True, name="inference-server2")

        if len(ic) < 1:
            return StepFail(results={"reason": "inference-server image not available"})


    # start inference server        
    cmd = ["bash", str(Vars.inference_server_dir / "start_pod.sh")]
    run_cmd_live(cmd, wait=False, printout=False)

    max_tries = 10
    tries = 0
    while not check_service(f"{Vars.inf_server_url}/state", 200):
        print("checking inference server state")
        if tries >= max_tries:
            return StepFail(results={"reason": "inference server not available"})
        sleep(10)
        
        tries += 1

    return StepSuccess(results={**prev_result.results})

    
def step4(prev_result: StepResult):
    """
    Build and start orchestrator
    """
    if not prev_result.success:
        raise StepFailedError(prev_result)
    
    if check_service(f"{Vars.orchestrator_url}/health", 200):
        print("Orchestrator OK")
        return StepSuccess(results={**prev_result.results})


    return StepSuccess(results={**prev_result.results})


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

    # r = requests.post(url=f"{Vars.inf_server_url}/remove_dd", json={"method": "all"})
    # rem = r.json()["remaining"]
    # if len(rem) > 0:
    #     return StepFail(results={"reason": f"failed to remove all detectors {rem}"})

    total_regimes =  len(EXPERIMENT_STATE["waiting"]["regime"])
    total_subsets = len(EXPERIMENT_STATE["waiting"]["subset"])
    total_iterations = total_subsets * total_regimes
    current_iteration = 1

    for i_subset, subset in enumerate(EXPERIMENT_STATE["waiting"]["subset"]):
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

        ready = False
        while not ready:
            rh = requests.get(url=f"{Vars.orchestrator_url}/monitor/current")
            history = rh.json()
            current_event_id = history["current_event"]["id"]
            current_workflow = history["current_workflow"]["workflow"]
            state = history["workflow_state"]

            if state == "failed":
                return StepFail("Training the model failed")

            training_ready = current_event_id == event_id and current_workflow == "Deployment workflow" and state == "complete"

            # check on inference server
            r = requests.get(url=f"{Vars.inf_server_url}/state")
            inference_server_ready = r.json()["state"] == "CoordinatorState.AVAILABLE"
        
            ready = training_ready and inference_server_ready
            
            print(f"Training: {training_ready}, inference server: {inference_server_ready}")
            sleep(5)


        print("Model deployed")

        r = requests.get(url=f"{Vars.inf_server_url}/available_models")
        cur_uuid = str(list(r.json()["available"].keys())[0])


        data = {"uuid": cur_uuid, "configs": methods_settings}
        r = requests.post(url=f"{Vars.inf_server_url}/add_plugins", json=data)
        print(r.json())

        for i, s in enumerate(r.json()["success"]):
            if not s:
                return StepFail(results={"reason": f"failed to add detector {methods_settings[i]}"})
        

        for i_regime, drift_regime in enumerate(EXPERIMENT_STATE["waiting"]["regime"]):
            print(f"\nROUND {current_iteration}/{total_iterations}: subset {i_subset + 1}/{total_subsets}, regime {i_regime + 1}/{total_regimes}\n")

            # set online data and drift function for inference client
            on_data = prev_result.results["data_source"] + str(subset) + "/online/"

            r = requests.post(url=f"{Vars.inf_client_url}/set_drift_func", json=drifts[drift_regime])
            
            if not r.json()["found"]:
                return StepFail(results={"reason": f"failed to set drift function {drifts[drift_regime]}"})

            r = requests.post(url=f"{Vars.inf_client_url}/set_data_source", json={"location": on_data})
            if not r.json()["found"]:
                return StepFail(results={"reason": f"failed to set data source {on_data}"})


            detections = {
                "conformal": False,
                "mmd": False,
            }
            def stopping_condition(data):
                if isinstance(data, str):
                    data = json.loads(data)
                if data["name"] in detections.keys():
                    if not detections[data["name"]] and data["is_drift"]:
                        detections[data["name"]] = True 
                    
                return all(detections.values()) # wait until all detectors have fired

            stop_event.clear()
            print(f"stopping condition reset({stop_event.is_set()}):", detections)

            server = create_server(stopping_condition)

            Thread(target=server.serve_forever, daemon=True).start()

            print("Waiting for notification...")

            cmd = ["bash", str(Vars.inference_client_dir / "send_n_samples.sh"), "10", cur_uuid]
            while not stop_event.is_set():
                
                # send samples
                returncode, buffer = run_cmd_live(cmd, printout=False)
                if returncode != 0:
                    print(buffer)
                    print("Curl exited with non-zero exit code")
                    break

                r = requests.get(f"{Vars.inf_client_url}/is_out_of_samples")
                if r.json()["out_of_samples"]:
                    csi = r.json()["current_sample_index"]
                    ts = r.json()["total_samples"]

                    r = requests.post(url=f"{Vars.inf_server_url}/sync_plugins", json={"uuid": cur_uuid}, timeout=None)

                    stop_event.set()

                sleep(0.1)

            print("Stopping condition achieved.")
            server.shutdown()
            del server
            
            # collect trace log and reset drift detection
            r = requests.post(url=f"{Vars.inf_server_url}/reset_plugin_log", json={"uuid": cur_uuid})
            clog = r.json()["log_file"]

            r = requests.post(url=f"{Vars.inf_server_url}/reset_plugins", json={"uuid": cur_uuid})

            EXPERIMENT_STATE["completed"]["associated_log_files"].append(clog)
            EXPERIMENT_STATE["completed"]["regime"].append(drift_regime)
            EXPERIMENT_STATE["completed"]["subset"].append(subset)

            # data = {"create_new_log": True}
            # r = requests.post(url=f"{Vars.inf_server_url}/reset_dd", json=data)

            current_iteration += 1

        r = requests.post(f"{Vars.inf_server_url}/remove_deployment", json={"uuid": cur_uuid})
        if r.json()["remaining"] != []:
            StepFail()

    return StepSuccess(results={
        **prev_result.results,
        "associated_log_files": EXPERIMENT_STATE["completed"]["associated_log_files"],
        "regimes": EXPERIMENT_STATE["completed"]["regime"],
        "subsets": EXPERIMENT_STATE["completed"]["subset"]

        })


def step6(prev_result: StepResult):
    if not prev_result.success:
        raise StepFailedError(prev_result)

    results = prev_result.results

    filenames = [Path(f)  for f in results.get("associated_log_files", [])]
    regs = results.get("regimes")
    subs = results.get("subsets")

    if len(regs) != len(subs) or len(subs) != len(filenames):raise
    # return StepFail(results={
    #     "reason": f"Length mismatch: regs={len(regs)}, subs={len(subs)}, filenames={len(filenames)}"
    #     })

    now = time.strftime("%Y%m%d_%H%M%S")
    data_directory = PROJECT_ROOT / "notebooks" / "data" / f"{now}"
    os.makedirs(data_directory, exist_ok=True)

    for src, reg, sbs in zip(filenames, regs, subs):
        dst = data_directory / f"{reg}_{sbs}_{src.name}"
        print("COPY", src, dst)

        shutil.copy(src, dst)





        



if __name__ == "__main__":
    
    drifts = {
        "gradual": {
            "drift_func_name": "CP2DriftFunc",
            "drift_func_kwargs": {
                "augmentation_func": "scratch_and_blur",
                "augmentation_kwargs": {
                    "n_scratches_range": [3,5],
                    "length_range": [0.3, 0.6],
                    "width_range": [0.01, 0.02],
                    "angle_range": [0, 6.2832],
                    "circularity_range": [0.0, 1.0],
                    "degree": 4
                },
                "parameter_name": "sigma",
                "param_values": [0.7, 1.2],
                "param_cps": [0.3, 0.7],
                "class_ratio": 0.5,
                "replace_all_labels": True,
            }
        },
        "sudden": {
            "drift_func_name": "CP2DriftFunc",
            "drift_func_kwargs": {
                "augmentation_func": "scratch_and_blur",
                "augmentation_kwargs": {
                    "n_scratches_range": [3,5],
                    "length_range": [0.3, 0.6],
                    "width_range": [0.01, 0.02],
                    "angle_range": [0, 6.2832],
                    "circularity_range": [0.0, 1.0],
                    "degree": 4
                },
                "parameter_name": "sigma",
                "param_values": [0.7, 1.2],
                "param_cps": [0.3, 0.305],
                "class_ratio": 0.5,
                "replace_all_labels": True,
            }
        }
    }

    #     plugin_type: Literal["eval", "drift"]
    # name: str
    # parameters: dict
    # trace_fields: list[str]
    # notifiers: list[str] = Field(default_factory=list)
    # reference_window_update: int = Field(default=10_000)
    # reset_on_detection: bool = Field(default=False)



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
            "plugin_type": "drift",
            "name": "conformal",
            "parameters": {
                "quantile_probability":0.95,
            },
            "trace_fields": ["threshold", "stat"],
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

    EXPERIMENT_STATE = {
        "completed": {
            "subset": [],
            "regime": [],
            "associated_log_files": [],
            "combination": []
            },

        "waiting": {
            "subset": [],
            "regime": [],
        }  
    }

    regimes = ["gradual", "sudden"]
    subsets = list(range(10))

    # return from previous state
    os.makedirs(Vars.temp_dir, exist_ok=True)
    state_file = Vars.temp_dir / "state.json"
    if False:#os.path.exists(state_file):
        with open(state_file, "r") as f: EXPERIMENT_STATE = json.load(f)
    
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
    steps = [step0, step1, step2, step3, step4, step5, step6]

    previous = StepSuccess()

    for i, step in enumerate(steps):
        print(f"\nSTEP {i}\n")

        #try:
        stepresult = step(previous)
        print(f"STEP {i} result: {stepresult}")
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

            
