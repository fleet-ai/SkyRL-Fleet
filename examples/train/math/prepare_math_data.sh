#!/usr/bin/env bash
# Prepare the Hendrycks MATH (levels 3-5) dataset for SkyRL training.
#
# Downloads DigitalLearningGmbH/MATH-lighteval (ungated drop-in for the DMCA'd
# hendrycks/competition_math), filters to difficulty levels 3-5, extracts the
# boxed ground-truth answer, and writes train.parquet / validation.parquet in
# the same schema the `aime` env (Hendrycks-MATH grader) expects.
#
# Example:
#   DATA_DIR=~/data/math bash examples/train/math/prepare_math_data.sh
set -uxo pipefail

export DATA_DIR=${DATA_DIR:-"${HOME}/data/math"}
export LEVELS=${LEVELS:-"3 4 5"}

mkdir -p "${DATA_DIR}"

uv run --isolated --extra fsdp -m examples.train.math.math_dataset \
  --output-dir "${DATA_DIR}" \
  --levels ${LEVELS}
