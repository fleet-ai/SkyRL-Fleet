"""Workaround for the `False backend is not supported` bug in skypilot-nightly 1.0.0.dev20260519.

The CLI's `--docker` option has default=False (should be None), so the
`if backend_name is None` fallback never fires and sky launch errors out.
Calling sky.launch() directly via the Python API bypasses the CLI parsing.
"""

import os
import sys
import sky

YAML_PATH = "tasks/baseline-qwen35-9b-smoke.yaml"

task = sky.Task.from_yaml(YAML_PATH)

# Forward the env vars sky CLI would have set via --env.
env_keys = ("FLEET_API_KEY", "WANDB_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
for k in env_keys:
    v = os.environ.get(k)
    if not v:
        print(f"ERROR: {k} not set in environment", file=sys.stderr)
        sys.exit(1)
task.update_envs({k: os.environ[k] for k in env_keys})

print(f"Launching task '{task.name}' from {YAML_PATH}...")
print(f"Resource candidates: {len(task.resources)}")

request_id = sky.launch(
    task,
    cluster_name=task.name,
    retry_until_up=True,
)
print(f"sky.launch returned request_id: {request_id}")
# In skypilot client/server mode, sky.launch returns a request id; stream it.
result = sky.stream_and_get(request_id)
print(f"final result: {result}")
