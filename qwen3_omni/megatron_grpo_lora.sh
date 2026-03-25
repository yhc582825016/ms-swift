#!/usr/bin/env bash
# Megatron + GRPO + LoRA for Qwen3-Omni-30B-A3B-Instruct
# Recommended: 8 * 80GiB (or more). Adjust parallel sizes with your GPU topology.
cd /mnt/code/yehangcheng/github/ms-swift
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
ENABLE_AUDIO_OUTPUT=1 \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
megatron rlhf \
    --rlhf_type grpo \
    --model /opt/users/ye/checkpoints/Qwen3-Omni-30B-A3B-Instruct-220/checkpoint-44398 \
    --save_safetensors true \
    --merge_lora false \
    --context_parallel_size 1 \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 4 \
    --pipeline_model_parallel_size 1 \
    --dataset lmms-lab/multimodal-open-r1-8k-verified#1000 \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_r1v_acc format \
    --num_train_epochs 1 \
    --global_batch_size 32 \
    --micro_batch_size 1 \
    --steps_per_generation 1 \
    --num_generations 8 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_tensor_parallel_size 4 \
    --vllm_max_model_len 24576 \
    --max_length 24576 \
    --max_completion_length 12000 \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --lr 5e-5 \
    --bf16 true \
    --beta 0.001 \
    --importance_sampling_level sequence \
    --epsilon 3e-4 \
    --epsilon_high 4e-4 \
    --dynamic_sample false \
    --overlong_filter true \
    --loss_type grpo \
    --sleep_level 2 \
    --offload_model true \
    --offload_bridge false \
    --offload_optimizer true \
    --logging_steps 1 \
    --recompute_granularity selective \
    --finetune \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim \
    --no_save_rng \
    --attention_backend flash \
    --temperature 1.0 \
    --system examples/train/grpo/prompt.txt \
    --padding_free true \
    --sequence_parallel true \
    --log_completions true \
    --report_to tensorboard \
    --output_dir megatron_output/Qwen3-Omni-30B-A3B-Instruct-grpo-lora
