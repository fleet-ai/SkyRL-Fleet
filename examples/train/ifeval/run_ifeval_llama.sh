set -x

# Colocated GRPO RLVR training for Llama-3.1-8B-Instruct on IFEval.
# IFEval uses deterministic, rule-based rewards (partial credit 0–1 per prompt).
#
# Steps to run:
#   1. uv run examples/train/ifeval/ifeval_dataset.py --output_dir $HOME/data/ifeval
#   2. export WANDB_API_KEY=<key>
#   3. bash examples/train/ifeval/run_ifeval_llama.sh
#
# Override defaults, e.g.: NUM_GPUS=4 bash examples/train/ifeval/run_ifeval_llama.sh
# For 4 GPUs with TP=2, set: NUM_GPUS=4 and add
#   generator.inference_engine.tensor_parallel_size=2 \
#   generator.inference_engine.num_engines=2

: "${DATA_DIR:="$HOME/data/ifeval"}"
: "${NUM_GPUS:=8}"
: "${LOGGER:=wandb}"
: "${INFERENCE_BACKEND:=vllm}"

uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_base \
  trainer.algorithm.advantage_estimator="grpo" \
  "data.train_data=['${DATA_DIR}/train.parquet']" \
  "data.val_data=['${DATA_DIR}/validation.parquet']" \
  trainer.policy.model.path="meta-llama/Llama-3.1-8B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.placement.policy_num_gpus_per_node="$NUM_GPUS" \
  trainer.placement.ref_num_gpus_per_node="$NUM_GPUS" \
  generator.inference_engine.num_engines="$NUM_GPUS" \
  generator.inference_engine.tensor_parallel_size=1 \
  trainer.policy_mini_batch_size=64 \
  trainer.train_batch_size=256 \
  trainer.micro_forward_batch_size_per_gpu=32 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.max_prompt_length=1024 \
  generator.max_input_length=1024 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.01 \
  trainer.ckpt_interval=100000 \
  trainer.epochs=50 \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=true \
  environment.env_class=ifeval \
  generator.n_samples_per_prompt=8 \
  generator.inference_engine.gpu_memory_utilization=0.7 \
  generator.sampling_params.temperature=0.6 \
  generator.sampling_params.top_p=0.95 \
  trainer.logger="$LOGGER" \
  trainer.project_name="skyrl" \
  trainer.run_name="ifeval_llama31_8b" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/ifeval_llama31_8b_ckpt" \
  trainer.eval_batch_size=100 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  $@
