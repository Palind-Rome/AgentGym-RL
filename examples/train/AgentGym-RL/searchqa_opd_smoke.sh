#!/usr/bin/env bash
set -euo pipefail
set -x

# A tiny OPD dataflow smoke test for a single consumer GPU.
# It checks: env rollout -> old logprobs -> ref-as-teacher logprobs -> OPD actor update.
# Start the AgentGym environment server before running this script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}/AgentGym-RL"

export VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE=offline

TASK_NAME="${TASK_NAME:-searchqa}"
ENV_SERVER_URL="${ENV_SERVER_URL:-http://127.0.0.1:36005}"
MODEL_PATH="${MODEL_PATH:-models/Qwen2.5-0.5B-Instruct}"
EXP_NAME="${EXP_NAME:-searchqa_opd_smoke}"
SAVE_DIR="${SAVE_DIR:-saves/${EXP_NAME}}"

mkdir -p "${SAVE_DIR}"

python3 -m verl.agent_trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.distillation.enabled=True \
    algorithm.distillation.teacher_model.path=null \
    algorithm.distillation.loss_mode=pg_reverse_kl \
    algorithm.distillation.loss_coef=1.0 \
    algorithm.distillation.mask_key=action_mask \
    algorithm.distillation.use_task_policy_loss=True \
    algorithm.rounds_ctrl.type=fixed \
    algorithm.rounds_ctrl.rounds=1 \
    data.train_file=AgentItemId/${TASK_NAME}_train.json \
    data.train_batch_size=1 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    actor_rollout_ref.agentgym.task_name=${TASK_NAME} \
    actor_rollout_ref.agentgym.env_addr=${ENV_SERVER_URL} \
    actor_rollout_ref.agentgym.timeout=120 \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=2048 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
    actor_rollout_ref.rollout.max_num_seqs=8 \
    actor_rollout_ref.rollout.max_tokens=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.rollout_log_dir=${SAVE_DIR}/executer_logs \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.default_local_dir=${SAVE_DIR} \
    trainer.project_name=agentgym_rl_opd \
    trainer.experiment_name=${EXP_NAME} \
    'trainer.logger=[console]' \
    trainer.save_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1
