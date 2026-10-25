# Example usage

This walkthrough uses the local demonstration stack described in
[`ml-platform/README.md`](ml-platform/README.md). It creates LakeFS data,
trains a small CPU model, deploys it, and sends predictions. It writes to the
LakeFS repository configured in `example/test_data_config.json`, PostgreSQL,
and MLflow; use it only with the local demo services.

## 1. Generate simple data

From `ml-platform/`, generate the data described by the existing test-data
configuration kept alongside this walkthrough:

```bash
ML_PLATFORM_DATA_DIR="$(cd .. && pwd)" \
  pyenv exec python -m ml_platform.data_generation.generate_datasets \
  --config ../example/test_data_config.json
```

The final `exp stage done, artifact in ...` line is the generated experiment
location. The training dataset is its first offline split; append
`0/offline/`. For example, set the location printed by your run:

```bash
export DATASET_LOCATION="devtest12/<generated-experiment-branch>/expdata/0/offline/"
export INFERENCE_DATASET="${DATASET_LOCATION%offline/}online/"
```

You can view this data LakeFS from the repository at http://localhost:8000.

## 2. Train, validate, and deploy a small model

Submit the training request in
[`example/training_request.json`](example/training_request.json).
`deploy_criteria: "none"` is intentional for this minimal walkthrough: it
deploys after training and validation regardless of the validation score. Use
`"validated"` when deployment must depend on a passing validation rule.

The template contains `${DATASET_LOCATION}` in the two fields that need the
location generated in step 1. This command expands that environment variable
without writing a modified configuration file:

```bash
pyenv exec python -c 'import os, pathlib; print(os.path.expandvars(pathlib.Path("../example/training_request.json").read_text()))' \
  | curl -sS -X POST http://localhost:8014/events/manual-start \
      -H 'Content-Type: application/json' -d @-
```

Copy the returned `event_id` into `EVENT_ID`, then inspect the event while it
runs and after it completes:

```bash
export EVENT_ID="<event-id>"
curl -sS "http://localhost:8014/events/${EVENT_ID}"
curl -sS "http://localhost:8014/events/${EVENT_ID}/history"
```

The final event result contains `model_uri`, validation output, `deployed`,
and, on successful deployment, `uuid`. The MLflow UI is available at
<http://localhost:5000>; inspect the `example-usage` run and the
`example_simple_model` registered model there.

After the event is `complete` and its result says `"deployed": true`, obtain
the inference deployment ID:

```bash
export DEPLOYMENT_ID="$(curl -sS "http://localhost:8014/events/${EVENT_ID}" | \
  pyenv exec python -c 'import json, sys; print(json.load(sys.stdin)["result"]["uuid"])')"
```

## 3. Inspect available inference deployments

```bash
curl -sS http://localhost:8012/available_models
```

The returned map associates deployment UUIDs with model names. Deployments are
in-memory: restarting the inference server removes them.

## 4. Send predictions

The inference client is the easiest path because it reads the generated online
split from LakeFS and serializes its samples correctly. Configure its no-op
drift function and data source, then send samples to the deployment:

```bash
curl -sS -X POST http://localhost:8013/set_drift_func \
  -H 'Content-Type: application/json' \
  -d '{"drift_func_name":"DriftFunc","drift_func_kwargs":{}}'

curl -sS -X POST http://localhost:8013/set_data_source \
  -H 'Content-Type: application/json' \
  -d @- <<JSON
{
  "location": "${INFERENCE_DATASET}",
  "dataset_description": {
    "sample_column": ["X", "bytes"],
    "sample_shape": [8],
    "target_column": ["y", "int"],
    "target_shape": [1],
    "order_column": ["write_timestamp", "datetime"],
    "target_interp": "classification"
  }
}
JSON

curl -sS -X POST http://localhost:8013/send_n_samples \
  -H 'Content-Type: application/json' \
  -d "{\"uuid\": \"${DEPLOYMENT_ID}\", \"n\": 10, \"labels\": true}"
```

For a direct request to the inference server, encode a NumPy array with the
project helper before calling `/predict`:

```bash
DEPLOYMENT_ID="$DEPLOYMENT_ID" pyenv exec python - <<'PY'
import os
import numpy as np
import requests

from ml_platform.shared.data_processing.data_processing import encode_for_json

response = requests.post(
    "http://localhost:8012/predict",
    json={
        "deployment_uuid": os.environ["DEPLOYMENT_ID"],
        "X": encode_for_json(np.zeros(8)),
        "y": 0,
    },
    timeout=30,
)
response.raise_for_status()
print(response.json())
PY
```
