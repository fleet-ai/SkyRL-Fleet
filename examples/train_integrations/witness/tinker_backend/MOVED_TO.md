# MOVED (2026-08-04) → github.com/fleet-ai/arc-witness-train

The tinker track (train_sft.py, main_witness_tinker.py, frontier.py,
_grpo_utils.py) moved to the dedicated training repo `arc-witness-train`,
seeded fresh-copy from this branch @ 3057da95 with md5 byte-identity verified
against the deployed cluster copies. The .py files here are raising stubs.

The slurm full-param track (../env.py, ../env_agent.py, ../sft_trainer_v3.py,
../entrypoints/) STAYS HERE — primitivbench imports it by module path and
published-run provenance stamps dereference into this fork. History of the
moved files: `git log <tag pre-restructure-2026-08> -- $(dirname $0)`.
