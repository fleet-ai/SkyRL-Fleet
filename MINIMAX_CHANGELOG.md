# MiniMax M2.7 Training — Changelog

All changes made to get MiniMax M2.7 (230B/10B MoE) training running on the RunPod Slurm cluster.

## Dependency Fixes (fleet-minimax-m27-extra-setup.sh)

1. **torch 2.10 → 2.11** — vLLM 0.22 requires torch 2.11. Installed with `--no-deps` to avoid breaking shared venv.
2. **torchvision 0.25 → 0.26** — torch 2.11 requires matching torchvision. Old version caused `operator torchvision::nms does not exist`.
3. **NCCL 2.27 → 2.28.9** — torch 2.11 uses `ncclDevCommCreate` (NCCL 2.28+ API). `--no-deps` left old NCCL from torch 2.10.
4. **vLLM 0.18 → 0.22 cu129** — MiniMax M2.7 needs vLLM 0.22. Default PyPI wheel is cu130 (`libcudart.so.13` missing on CUDA 12.8 nodes). Installed cu129 wheel from GitHub releases.
5. **flashinfer-jit-cache mismatch** — vLLM 0.22 upgrades flashinfer-python to 0.6.11 but base venv's jit-cache stays at 0.6.6. Upgraded from flashinfer.ai/whl/cu129.
6. **FLASHINFER_DISABLE_VERSION_CHECK=1** — flashinfer-jit-cache 0.6.12+cu129 still doesn't match flashinfer-python 0.6.11.post2 exactly (CUDA suffix). Bypass in fleet-common-run.sh.
7. **CUDA 12.9 toolkit** — DeepGemm JIT (FP8 MoE kernels) needs nvcc 12.9. RunPod containers have legacy apt source that conflicts with cuda-keyring. Fixed by removing legacy source before installing.
8. **flash-attn** — installed from source for torch 2.11 compat.

## SkyPilot / Slurm Fixes (setup-slurm.sh, fleet-launch.sh)

9. **provision_timeout: -1** — SkyPilot's default 2-min timeout cancels Slurm PD jobs and retries. Set to -1 (infinite) so SkyPilot waits for Slurm's scheduler.
10. **--down auto-teardown** — Re-enabled after provision_timeout fix. Tears down cluster after job finishes. Removed --keep-cluster flag.
11. **Non-root user support** — Created Linux user on controller + all compute nodes with consistent UID, SSH key, passwordless sudo. Shows username in squeue instead of root.
12. **sq alias** — `squeue` with wide job name format, added to /etc/profile.d/.

## Training Config (fleet-minimax-m27-run.sh, YAML)

13. **Dynamic batch sizing** — `train_batch_size` and `policy_mini_batch_size` derived from GPU count (micro_train_batch_size_per_gpu=1 × num_gpus).
14. **TP=4** — 4 GPUs per vLLM engine, 8 engines on 32 GPUs.
15. **FSDP2 + CPU optimizer offload** — Required for 230B model.
16. **128K context** — MAX_INPUT_LENGTH=128000.

## Infrastructure Fixes (fleet-common-setup.sh, fleet-common-run.sh)

17. **Model pre-download to NFS** — Downloads model once on head node to /workspace/hf_cache during setup. All nodes read from NFS at runtime instead of 4 independent HF downloads.
18. **HF cache permissions** — chmod a+rwX with || true (shared cache, other users' files not chown-able).
19. **MAX_TASKS wiring** — Connected MAX_TASKS env var to prepare_dataset.py --max-tasks flag (was dead code).
20. **CI eval_before_train=false** — CI 35B tool-use YAML was running 20h eval before training.

## Colocated Training Docs (CLAUDE.md)

21. **Memory model** — Documented that vLLM and FSDP alternate on same GPUs (never simultaneous). gpu_memory_utilization is vLLM-phase only.

## Known Issues

- **--down auto-cleanup fails when job crashes during early init** — SkyPilot job status stays PENDING in jobs.db, Skylet never detects idle. Filed: https://github.com/skypilot-org/skypilot/issues/9815
- **NFS stale file handles** — Failed job cleanup can leave stale dirs that block next launch.
- **Node restarts wipe users** — useradd is not persistent on RunPod containers. Must re-run setup-slurm.sh after node restart.
