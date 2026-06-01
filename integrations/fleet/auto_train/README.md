# auto-train

Polls Fleet `task_projects` every 30 min via GH Actions, smoke-tests new
datasets, launches single-epoch CI training, posts to Slack.

## Flow

```
cron → discover (Supabase) → diff vs S3 state → smoke (Fleet SDK)
     → export to S3 → sky launch CI YAML → save ckpt to S3 → Slack
```

## What runs where

| Layer    | Path |
| -------- | ---- |
| Schedule | `.github/workflows/auto-train.yaml` (cron `*/30 * * * *`, 6h timeout) |
| Trigger  | `integrations/fleet/auto_train/` (`python -m ... trigger`) |
| Training | `tasks/openenv-fleet-grpo-{tu,bu,cu}-ci.yaml` + `scripts/fleet-*-run.sh` |

## S3 layout

| Purpose      | Path |
| ------------ | ---- |
| Input tasks  | `s3://fleet-internal-datasets/{key}/openenv/all_{modality}.json` |
| Seen state   | `s3://fleet-internal-datasets/.auto_train_state.json` |
| Checkpoints  | `s3://skyrl-checkpoints/{project}/{model}/{run_name}/global_step_N/` |
| Eval results | `s3://skyrl-trajectories/evals/{run_name}/global_step_N/` |

## Modality rules

- `task_modality=tool_use` → tu (Qwen3.5-35B)
- `task_modality=browser_use` → bu (Qwen3.5-9B VL)
- `task_modality=computer_use` AND env starts with `fos-` → cu (Qwen3.5-9B VL, `image_type=mcp`)
- `task_modality=computer_use` otherwise → treat as bu

## Operate

```
gh workflow run auto-train.yaml -f max_datasets=1    # run now
python -m integrations.fleet.auto_train status       # processed pairs
gh workflow disable auto-train.yaml                  # turn off
```

Slack channel: `#fleet-training-runs` (Fleet Training Bot, `chat:write`).

## Secrets (14, in `fleet-ai/skyrl-fleet` GH settings)

`FLEET_API_KEY`, `WANDB_API_KEY`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
`GCP_SA_KEY`, `RUNPOD_API_KEY`, `RUNPOD_CONFIG_TOML`, `RUNPOD_SSH_KEY`,
`SKY_CONFIG_YAML`, `SLURM_CONFIG`, `SSH_ID_ED25519`, `SLACK_BOT_TOKEN`,
`SUPABASE_URL`, `SUPABASE_KEY`.
