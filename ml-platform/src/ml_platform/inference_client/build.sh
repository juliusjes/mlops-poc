
#!/usr/bin/env bash
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

cd "$script_dir/../.."

echo "Building inference-client container in: $PWD"

podman build  -f ./src/Dockerfile-inference-client.Dockerfile -t inference-client .