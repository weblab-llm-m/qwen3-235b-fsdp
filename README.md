# Qwen3-235B-A22B-FSDP

A distributed training framework specialized for **selective expert training** of Qwen3-235B-A22B (Mixture-of-Experts). Supports distributed training at 64 GPU scale via FSDP (Fully Sharded Data Parallel) with real-time weight synchronization to vLLM servers.

## Features

- **Selective Expert Training**: Train only a subset of MoE experts while freezing the rest
- **Multiple Training Pipelines**: Full-parameter GRPO / LoRA GRPO / SFT / DPO
- **Dual Judge Reward System**: Medical (accuracy) + SMART (helpfulness) LLM-as-judge
- **Reward Hacking Defense**: Medical score gating and penalty mechanisms
- **FSDP-vLLM Real-time Sync**: Stream weight updates to inference servers during training
- **3 RL Objective Functions**: GRPO / GDPO / GSPO
- **Merge Tools**: FSDP sharded → full conversion, expert bake-in, LoRA merge

## Directory Structure

```
qwen3-235b-fsdp/
├── grpo/
│   ├── full/                              # Full-parameter GRPO training
│   │   ├── conda/
│   │   │   └── setup_qwen_h100_one_shot.sh
│   │   ├── script/
│   │   │   ├── run_expert_fullparam_grpo.py
│   │   │   ├── run_vllm_qwen3_235B_servers_multi.sh
│   │   │   └── slurm_qwen3_235b_expert_grpo.sh
│   │   └── trl/
│   │       ├── vllm_client.py             # FSDP-compatible vLLM client
│   │       └── vllm_serve.py              # vLLM server with weight sync extension
│   │
│   └── lora/                              # LoRA GRPO training
│       ├── conda/
│       │   └── setup_qwen_h100_one_shot.sh
│       └── script/
│           ├── run_lora_grpo.py
│           ├── run_vllm_qwen3_235B_lora_server.sh
│           └── slurm_qwen3_235b_lora_grpo.sh
│
├── dpo/                                   # Direct Preference Optimization (DPO)
│   ├── run_expert_fullparam_dpo.py
│   └── slurm_qwen3_235b_expert_dpo.sh
│
├── sft/                                   # Supervised Fine-Tuning (SFT)
│   ├── conda/
│   │   └── setup_qwen_h100_one_shot.sh
│   └── script/
│       ├── run_expert_fullparam_sft.py
│       └── slurm_qwen3_235b_expert_sft.sh
│
└── merge/                                 # Model merging & consolidation
    ├── fsdp/
    │   ├── merge_experts_and_bake_base.py  # Bake experts into base model
    │   └── slurm_qwen3_235b_expert_merge.sh
    ├── lora/
    │   └── slurm_merge_loras.sh            # LoRA adapter merge
    └── pytorch/
        ├── fsdp_sharded_to_full_torchrun.py # Sharded → full checkpoint conversion
        └── slurm_qwen3_235b_expert_merge.sh
```

## Prerequisites

- **GPU**: NVIDIA H100 x 64 (8 nodes x 8 GPUs)
- **CUDA**: 12.4
- **Conda**: Miniconda 24.7.1
- **Python**: 3.10 (LoRA vLLM server environment only: 3.11)
- **SLURM**: Job scheduler
- **NCCL**: 2.22.3+
- **Network**: High-speed interconnect (InfiniBand / NVLink)

## Setup

Run the environment setup script in the `conda/` directory of each pipeline.

```bash
# For full-parameter GRPO / SFT / DPO (env name: qwen_expert_rl)
bash grpo/full/conda/setup_qwen_h100_one_shot.sh
# or equivalently:
bash sft/conda/setup_qwen_h100_one_shot.sh

# For LoRA GRPO (env name: qwen_lora_rl)
bash grpo/lora/conda/setup_qwen_h100_one_shot.sh
```

The script automatically performs:

1. Delete and recreate the environment (Python 3.10)
2. Install PyTorch (CUDA 12.4)
3. Install `transformers`, `datasets`, `peft`, `accelerate`, `trl[vllm]`
4. Build `deepspeed==0.17.5` (CUDA-enabled)
5. Compile `flash-attn`
6. Install `bitsandbytes>=0.43.3`, `wandb`, `hf_transfer`
7. Install `nvidia::cuda-toolkit=12.4` (nvcc)
8. Verify torch / CUDA / DeepSpeed

## Data Format

### Training Data (ChatML JSONL)

```jsonl
{"messages":[{"role":"user","content":"<user message>"},{"role":"assistant","content":"<assistant response>"}]}
{"messages":[{"role":"user","content":"<user message>"},{"role":"assistant","content":"<assistant response>"}]}
```

Each line is in JSON format with a `messages` field containing an array of `user` / `assistant` role messages. The `content` field is a text string, but dictionaries/arrays for multimodal use are also supported.

### Expert Mask JSON

A JSON file specifying which experts to train. Two formats are supported:

**Format 1: Per-layer expert specification**

```json
{
  "selected_layer_experts": [
    {"layer_idx": 0, "expert_id": 5},
    {"layer_idx": 0, "expert_id": 42},
    {"layer_idx": 1, "expert_id": 12},
    ...
  ]
}
```

**Format 2: Common expert IDs across all layers**

```json
{
  "selected_experts_unique": [5, 12, 42, 87]
}
```

Keys `selected_experts` or `picked_experts` are also accepted. Layers with an empty mask default to all experts enabled as a safety measure.

---

## 1. Supervised Fine-Tuning (SFT)

Performs supervised learning on selected experts. Supports offline pre-tokenization and length-bucketing for acceleration.

### Launch

```bash
sbatch sft/script/slurm_qwen3_235b_expert_sft.sh
```

### Basic Usage

```bash
accelerate launch \
    --num_machines 8 \
    --num_processes 64 \
    --mixed_precision bf16 \
    sft/script/run_expert_fullparam_sft.py \
    --model /path/to/Qwen3-235B-A22B-Instruct \
    --dataset /path/to/train.jsonl \
    --mask_json /path/to/selected_experts.json \
    --epochs 5 \
    --batch_size 1 \
    --lr 2e-5 \
    --seq_len 2048 \
    --optim adafactor \
    --save_dir /path/to/output \
    --wandb
```

### Full Argument Reference

#### Model & Data

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model` | str | env var `BASE_MODEL` | Path to base model |
| `--layers` | int | 94 | Number of model layers |
| `--experts` | int | 128 | Number of experts per layer |
| `--dataset` | str | env var `SPECIALIST_CHATML_PATH` | Path to ChatML JSONL training data |
| `--tokenized_dataset` | str | None | Path to pre-tokenized JSONL (takes priority when specified) |
| `--seq_len` | int | 2048 | Maximum sequence length |
| `--mask_json` | str | - | Expert selection mask JSON |

#### Training Control

| Argument | Type | Default | Description |
|---|---|---|---|
| `--epochs` | int | 0 | Number of epochs (epoch mode enabled when >0) |
| `--steps` | int | 1500 | Maximum steps (acts as cap in epoch mode) |
| `--batch_size` | int | 2 | Batch size per GPU |
| `--gradient_accumulation_steps` | int | 1 | Gradient accumulation steps |
| `--lr` | float | 2e-5 | Learning rate |
| `--weight_decay` | float | 0.0 | Weight decay |
| `--grad_clip` | float | 1.0 | Gradient clipping threshold |
| `--optim` | str | adafactor | Optimizer (`adafactor` / `adamw` / `zero_adamw`) |

#### Batching Optimization

| Argument | Type | Default | Description |
|---|---|---|---|
| `--packing` | flag | - | Pack multiple samples into seq_len to reduce padding and maximize GPU efficiency |
| `--bucket_buffer` | int | 1024 | Buffer size for length-bucketing (must be >= batch_size) |
| `--max_batch_len` | int | 0 | Max padded length per batch (0 to disable, must be <= seq_len) |
| `--num_workers` | int | 0 | Number of DataLoader workers |

#### Pre-tokenization

Offline pre-tokenization of training data for faster training startup.

```bash
python sft/script/run_expert_fullparam_sft.py \
    --model /path/to/model \
    --dataset /path/to/train.jsonl \
    --pretokenize \
    --pretokenize_out /path/to/tokenized.jsonl \
    --seq_len 2048
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--pretokenize` | flag | - | Run in pre-tokenization mode and exit |
| `--pretokenize_out` | str | None | Output path for tokenized results |

#### Save Options

| Argument | Type | Default | Description |
|---|---|---|---|
| `--save_dir` | str | expert_ckpts_qwen94x128_sft | Checkpoint save directory |
| `--save_updated_only` | flag | - | Save only trainable parameters (requires_grad=True) |
| `--save_legacy_only` | flag | - | Use only accelerator.save_state() |
| `--save_legacy_after_new` | flag | - | Also run accelerator.save_state() after new-format save |
| `--legacy_save_subdir` | str | legacy_accelerate_state | Subdirectory name for legacy save |

#### Logging & W&B

| Argument | Type | Default | Description |
|---|---|---|---|
| `--wandb` | flag | - | Enable W&B logging |
| `--wandb_project` | str | expert_fullparam_sft | W&B project name |
| `--wandb_run_name` | str | None | W&B run name |
| `--wandb_entity` | str | None | W&B entity |
| `--log_every` | int | 10 | Log output interval (steps) |

---

## 2. Direct Preference Optimization (DPO)

Performs full-parameter DPO training on selected experts and/or attention layers. Supports expert-only training via mask JSON and attention layer selective training. Includes multiple DPO fixes for MoE models (padding alignment, concat forward for routing consistency, checkpointed cross-entropy for OOM prevention).

### Launch

```bash
sbatch dpo/slurm_qwen3_235b_expert_dpo.sh
```

### Basic Usage

```bash
accelerate launch \
    --num_machines 6 \
    --num_processes 48 \
    --mixed_precision bf16 \
    dpo/run_expert_fullparam_dpo.py \
    --model /path/to/Qwen3-235B-A22B-Instruct \
    --dataset /path/to/train_dpo.jsonl \
    --attn_layers 70-93 \
    --attn_proj q_proj,k_proj,v_proj,o_proj \
    --beta 0.5 \
    --loss_type sigmoid \
    --epochs 1 \
    --batch_size 1 \
    --lr 5e-7 \
    --seq_len 4096 \
    --save_dir /path/to/output \
    --wandb_project expert_fullparam_dpo
```

### Data Format (DPO JSONL)

Each line contains a prompt with chosen and rejected responses:

```jsonl
{"prompt":"<prompt>","chosen":"<preferred response>","rejected":"<dispreferred response>"}
```

### Full Argument Reference

#### Model & Data

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model` | str | env var `BASE_MODEL` | Path to base model |
| `--layers` | int | 94 | Number of model layers |
| `--experts` | int | 128 | Number of experts per layer |
| `--dataset` | str | - | Path to DPO JSONL training data (prompt/chosen/rejected) |
| `--seq_len` | int | 4096 | Maximum sequence length |
| `--mask_json` | str | None | Expert selection mask JSON |

#### Attention Layer Training

| Argument | Type | Default | Description |
|---|---|---|---|
| `--attn_layers` | str | None | Attention layer range to train (e.g., `70-93`) |
| `--attn_proj` | str | None | Comma-separated projection names to train (e.g., `q_proj,k_proj,v_proj,o_proj`) |

#### DPO Parameters

| Argument | Type | Default | Description |
|---|---|---|---|
| `--beta` | float | 0.1 | DPO temperature parameter (controls deviation from reference policy) |
| `--loss_type` | str | sigmoid | DPO loss type (`sigmoid` / `hinge` / `ipo`) |
| `--label_smoothing` | float | 0.0 | Label smoothing factor |
| `--length_normalise` | flag | - | Normalize log-probabilities by sequence length |

#### Training Control

| Argument | Type | Default | Description |
|---|---|---|---|
| `--epochs` | int | 3 | Number of epochs |
| `--steps` | int | 0 | Maximum steps (0 for unlimited, uses epochs) |
| `--batch_size` | int | 1 | Batch size per GPU |
| `--gradient_accumulation_steps` | int | 4 | Gradient accumulation steps |
| `--seed` | int | 42 | Random seed |
| `--lr` | float | 5e-6 | Learning rate |
| `--weight_decay` | float | 0.01 | Weight decay |
| `--grad_clip` | float | 1.0 | Gradient clipping threshold |

#### Save & Logging

| Argument | Type | Default | Description |
|---|---|---|---|
| `--save_dir` | str | outputs/ckpts_dpo | Checkpoint save directory |
| `--save_every` | int | 0 | Save checkpoint every N steps (0 to save only at end) |
| `--log_every` | int | 1 | Log output interval (steps) |
| `--wandb_project` | str | - | W&B project name (empty to disable) |
| `--wandb_run_name` | str | - | W&B run name |
| `--activation_checkpointing` | flag | True | Enable activation checkpointing (default: enabled) |

---

## 3. Full-Parameter GRPO Training

Updates full parameters of selected experts via reinforcement learning (GRPO/GDPO/GSPO). Integrates a dual judge reward system using LLM-as-judge and real-time inference via vLLM.

### Setup: install patched TRL files

Full-parameter GRPO requires the two patched TRL files in `grpo/full/trl/` to be copied over the installed TRL package in the conda environment (they replace `trl/scripts/vllm_serve.py` and `trl/extras/vllm_client.py`):

```bash
conda activate qwen_expert_rl

# Adjust the python3.x directory to match your environment's Python version
cp grpo/full/trl/vllm_serve.py  "${CONDA_PREFIX}/lib/python3.10/site-packages/trl/scripts/vllm_serve.py"
cp grpo/full/trl/vllm_client.py "${CONDA_PREFIX}/lib/python3.10/site-packages/trl/extras/vllm_client.py"
```

This step must be repeated whenever the conda environment is recreated or TRL is reinstalled/upgraded.

### Launch

```bash
# 1. Start vLLM inference servers (2 nodes x 8 GPUs)
sbatch grpo/full/script/run_vllm_qwen3_235B_servers_multi.sh

# 2. Submit GRPO training job (8 nodes x 8 GPUs)
sbatch grpo/full/script/slurm_qwen3_235b_expert_grpo.sh
```

### Basic Usage

```bash
accelerate launch \
    --num_machines 8 \
    --num_processes 64 \
    --mixed_precision bf16 \
    grpo/full/script/run_expert_fullparam_grpo.py \
    --model /path/to/Qwen3-235B-A22B-Instruct \
    --dataset /path/to/train.jsonl \
    --mask_json /path/to/selected_experts.json \
    --epochs 5 \
    --batch_size 1 \
    --grpo_group_size 8 \
    --lr 5e-5 \
    --reward_mode gspo \
    --use_vllm \
    --vllm_base_urls "http://host1:8000,http://host2:8000" \
    --judge_backend cerebras+openrouter \
    --judge_model_name qwen-3-235b-a22b-instruct-2507 \
    --smart_backend openrouter \
    --centralize_actor_judge \
    --wandb
```

### Full Argument Reference

#### Model & Data

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model` | str | env var `BASE_MODEL` | Path to base model |
| `--layers` | int | 94 | Number of model layers |
| `--experts` | int | 128 | Number of experts per layer |
| `--dataset` | str | env var `SPECIALIST_CHATML_PATH` | Path to ChatML JSONL training data |
| `--seq_len` | int | 2048 | Maximum sequence length |
| `--mask_json` | str | - | Expert selection mask JSON |

#### Training Control

| Argument | Type | Default | Description |
|---|---|---|---|
| `--grpo_group_size` | int | 4 | GRPO group size (number of candidate responses per prompt) |
| `--epochs` | int | 0 | Number of epochs (epoch mode when >0) |
| `--steps` | int | 1500 | Maximum steps |
| `--batch_size` | int | 2 | Batch size per GPU |
| `--lr` | float | 5e-5 | Learning rate |
| `--weight_decay` | float | 0.0 | Weight decay |
| `--grad_clip` | float | 1.0 | Gradient clipping threshold |
| `--max_consecutive_nan` | int | 10 | Number of consecutive NaN/Inf losses before abort |

#### Attention-Only Training

A mode that freezes experts (FFN) and trains only Attention layers (Q/K/V/O). Adapts attention patterns while preserving language capabilities.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--train_attention_only` | flag | - | Enable Attention-Only training |
| `--attention_layer_start` | int | 0 | Start of trainable layer range (inclusive) |
| `--attention_layer_end` | int | -1 | End of trainable layer range (exclusive, -1 for all layers) |

```bash
# Example: Train only Attention in the latter half of layers (47-93)
python run_expert_fullparam_grpo.py \
    --train_attention_only \
    --attention_layer_start 47 \
    --attention_layer_end 94 \
    ...
```

#### Generation Parameters

| Argument | Type | Default | Description |
|---|---|---|---|
| `--gen_max_new_tokens` | int | 192 | Maximum tokens for candidate responses |
| `--gen_temperature` | float | 0.7 | Sampling temperature |
| `--gen_top_p` | float | 0.9 | Top-p (nucleus) sampling |
| `--gen_repetition_penalty` | float | 1.05 | Repetition penalty |

#### Reward Mode

| Argument | Type | Default | Description |
|---|---|---|---|
| `--reward_mode` | str | grpo | Reward mode (`grpo` / `gdpo` / `gspo`) |

**GRPO** (Group Relative Policy Optimization):
- Token-level PPO-clip + intra-group normalization

| Argument | Type | Default | Description |
|---|---|---|---|
| `--grpo_clip_eps` | float | 0.2 | PPO-clip epsilon |
| `--grpo_logratio_clip` | float | 20.0 | Log-ratio clamp range (prevents exp() overflow) |
| `--grpo_epochs` | int | 1 | Inner epochs (μ in the paper, typically 1 in practice) |

**GDPO** (Decoupled Group Policy Optimization):
- Token-level PPO-clip + decoupled normalization

| Argument | Type | Default | Description |
|---|---|---|---|
| `--gdpo_eps` | float | 1e-6 | Epsilon to prevent division by zero in normalization |
| `--gdpo_whiten_global` / `--no_gdpo_whiten_global` | flag | True | Batch-wide advantage normalization (Eq.6) |

**GSPO** (Sequence-level Group Policy Optimization):
- Sequence-level PPO-clip

| Argument | Type | Default | Description |
|---|---|---|---|
| `--gspo_clip_eps_low` | float | 3e-4 | Left clip epsilon (ratio >= 1-eps) |
| `--gspo_clip_eps_high` | float | 4e-4 | Right clip epsilon (ratio <= 1+eps) |
| `--gspo_logratio_clip` | float | 20.0 | Log-ratio clamp range |
| `--gspo_epochs` | int | 4 | Inner epochs (4 mini-batches in the paper) |

#### Reward Scaling

| Argument | Type | Default | Description |
|---|---|---|---|
| `--reward_scale` | float | 1.0 | Reward scaling factor |
| `--reward_clip` | float | 1.0 | Reward clipping range |
| `--ema_alpha` | float | 0.05 | EMA smoothing factor for reward baseline |
| `--reward_cache` | str | judge_score_cache.jsonl | Cache file for Medical judge scores |

#### Medical Judge Settings

The Medical judge evaluates medical accuracy of responses with scores of `{0, 0.5, 1}`.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--judge_backend` | str | local | Judge backend (see table below) |
| `--judge_model_name` | str | None | Judge model name |
| `--judge_fallback_model_name` | str | None | Fallback model name |
| `--judge_api_base` | str | None | API base URL |
| `--judge_api_key` | str | None | API key |
| `--judge_max_new_tokens` | int | 2048 | Maximum generation tokens for judge |
| `--judge_temperature` | float | 0.0 | Judge sampling temperature |
| `--judge_max_retries` | int | 1000 | Maximum retry count |
| `--judge_connect_timeout` | float | 10.0 | HTTP connection timeout (seconds) |
| `--judge_read_timeout` | float | 60.0 | HTTP read timeout (seconds) |
| `--judge_retry_sleep` | float | 1.0 | Base retry sleep (seconds) |
| `--judge_retry_sleep_max` | float | 60.0 | Maximum retry sleep (seconds) |
| `--judge_http_retries` | int | 8 | urllib3 retry count |
| `--judge_http_backoff` | float | 0.5 | urllib3 backoff factor |
| `--judge_parallel_workers` | int | 0 | Number of parallel workers (0=auto) |

**Judge Backend Options:**

| Backend | Description |
|---|---|
| `local` | Judge using local model |
| `openrouter` | OpenRouter API |
| `openai` | OpenAI API |
| `cerebras` | Cerebras API |
| `cerebras+openrouter` | Cerebras primary, fallback to OpenRouter on failure |
| `openrouter+cerebras` | OpenRouter primary, fallback to Cerebras on failure |

#### SMART Judge Settings

The SMART judge evaluates helpfulness, structure, and clarity of responses with integer scores from `0~10`. If omitted, the same backend as the Medical judge is used.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--smart_backend` | str | None (same as judge_backend) | SMART judge backend |
| `--smart_model_name` | str | None | SMART judge model name |
| `--smart_fallback_model_name` | str | None | SMART fallback model name |
| `--smart_api_base` | str | None | SMART API base URL |
| `--smart_api_key` | str | None | SMART API key |
| `--smart_max_new_tokens` | int | None (same as judge) | Maximum generation tokens |
| `--smart_temperature` | float | None (same as judge) | Sampling temperature |
| `--smart_max_retries` | int | None (same as judge) | Maximum retry count |
| `--smart_parallel_workers` | int | 0 | Number of parallel workers (0=auto) |

#### SMART Reward Shaping

Integrates SMART scores with Medical scores to compute the final reward. Includes mechanisms to prevent reward hacking (responses that are medically inaccurate but score high on SMART).

```
Final reward = medical_score + smart_add
smart_add = (SMART_score / smart_max_score) * smart_reward_weight
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--smart_reward_cache` | str | smart_score_cache.jsonl | SMART score cache file |
| `--smart_max_score` | float | 10.0 | Maximum SMART score (for normalization) |
| `--smart_reward_weight` | float | 0.2 | SMART reward weight |
| `--smart_gate_mode` | str | soft | Gating mode (`soft` / `hard` / `none`) |
| `--smart_gate_threshold` | float | 0.5 | Hard gate threshold |
| `--smart_reward_cap` | float | 0.2 | SMART reward cap (<=0 to disable) |
| `--smart_hack_penalty` | float | 0.2 | Penalty on hacking detection (0 to disable) |
| `--smart_hack_med_le` | float | 0.0 | Hacking detection: Medical score threshold (<=) |
| `--smart_hack_smart_ge` | float | 8.0 | Hacking detection: SMART score threshold (>=) |

**Gating Modes:**
- `soft`: `smart_add *= medical_score` (continuously weighted by Medical score)
- `hard`: SMART reward added only when Medical >= threshold
- `none`: No gating (SMART reward always added)

**Hacking Detection:** Penalty applied when Medical <= `smart_hack_med_le` AND SMART >= `smart_hack_smart_ge`.

#### vLLM Integration

Generates candidate responses via vLLM servers during training with real-time FSDP weight synchronization.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--use_vllm` | flag | - | Enable vLLM integration |
| `--vllm_base_url` | str | None | vLLM server URL (single server) |
| `--vllm_base_urls` | str | None | Comma-separated multiple server URLs |
| `--vllm_update_every` | int | 1 | Sync weights every N optimization steps |
| `--vllm_force_n1` | flag | - | Emulate n>1 as repeated n=1 calls (for debugging) |
| `--vllm_connection_timeout` | int | 240 | Connection timeout (seconds) |

#### Centralize Actor-Judge

| Argument | Type | Default | Description |
|---|---|---|---|
| `--centralize_actor_judge` | flag | - | Rank 0 handles generation + judging for all ranks and distributes results (reduces inter-rank skew) |

#### Evaluation (Eval)

Runs periodic evaluation with temp=0 during training.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--eval_every` | int | 50 | Run evaluation every N optimization steps (0 to disable) |
| `--eval_size` | int | 100 | Number of evaluation samples |
| `--eval_max_new_tokens` | int | 2048 | Maximum generation tokens during evaluation |
| `--eval_temperature` | float | 0.0 | Evaluation sampling temperature |
| `--eval_top_p` | float | 1.0 | Evaluation top-p |
| `--eval_dataset` | str | None | Evaluation dataset path (uses training data if unspecified) |

#### Sample Output

Records prompts, responses, and scores in JSONL format during training.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--samples_jsonl_path` | str | samples.jsonl | Sample output path (can be relative to save_dir) |
| `--samples_jsonl_every` | int | 1 | Record every N steps (0 to disable) |
| `--samples_jsonl_overwrite` | flag | - | Overwrite file at start (default is append) |
| `--samples_jsonl_text_max_chars` | int | 0 | Maximum characters for text fields (0=unlimited) |

#### Save Options

| Argument | Type | Default | Description |
|---|---|---|---|
| `--save_dir` | str | expert_ckpts_qwen94x128_grpo | Checkpoint save directory |
| `--save_updated_only` | flag | - | Save only trainable parameters |
| `--save_legacy_only` | flag | - | Use only accelerator.save_state() |
| `--save_legacy_after_new` | flag | - | Also run legacy save after new-format save |
| `--legacy_save_subdir` | str | legacy_accelerate_state | Subdirectory name for legacy save |

#### Logging & W&B

| Argument | Type | Default | Description |
|---|---|---|---|
| `--wandb` | flag | - | Enable W&B logging |
| `--wandb_project` | str | expert_fullparam_grpo | W&B project name |
| `--wandb_run_name` | str | None | W&B run name |
| `--wandb_entity` | str | None | W&B entity |
| `--wandb_table_rows` | int | 4 | Number of batch rows to log in W&B tables |
| `--wandb_table_text_max_chars` | int | 2000 | Maximum characters for W&B table text cells (0=unlimited) |
| `--log_every` | int | 20 | Log output interval (steps) |
| `--debug_print` | flag | - | Enable debug output |

---

## 4. LoRA GRPO Training

Freezes the base model and updates only LoRA adapters via reinforcement learning. Under FSDP, LoRA parameters are set as `ignored_states` to exclude them from sharding, and gradients are manually synchronized via all-reduce.

### Launch

```bash
# 1. Start vLLM server (with LoRA runtime support)
sbatch grpo/lora/script/run_vllm_qwen3_235B_lora_server.sh

# 2. Submit LoRA GRPO training job
sbatch grpo/lora/script/slurm_qwen3_235b_lora_grpo.sh
```

### Basic Usage

```bash
accelerate launch \
    --num_machines 8 \
    --num_processes 64 \
    --mixed_precision bf16 \
    grpo/lora/script/run_lora_grpo.py \
    --model /path/to/Qwen3-235B-A22B-Instruct \
    --dataset /path/to/train.jsonl \
    --epochs 1 \
    --batch_size 2 \
    --grpo_group_size 4 \
    --lr 5e-5 \
    --reward_mode gspo \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj \
    --use_vllm \
    --vllm_base_urls "http://host1:8000,http://host2:8000" \
    --vllm_model_name qwen3-235b-base \
    --vllm_lora_name train-lora \
    --judge_backend openrouter \
    --judge_model_name qwen/qwen3-235b-a22b-2507 \
    --wandb
```

### Full Argument Reference

#### Model & Data

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model` | str | env var `BASE_MODEL` | Path to base model |
| `--dataset` | str | env var `SPECIALIST_CHATML_PATH` | Path to ChatML JSONL training data |
| `--seq_len` | int | 2048 | Maximum sequence length |

#### LoRA Settings

| Argument | Type | Default | Description |
|---|---|---|---|
| `--lora_r` | int | 16 | LoRA rank |
| `--lora_alpha` | int | 32 | LoRA alpha (scaling) |
| `--lora_dropout` | float | 0.05 | LoRA dropout rate |
| `--lora_target_modules` | str | None | Comma-separated target module names (auto-detected if omitted) |

#### Training Control

| Argument | Type | Default | Description |
|---|---|---|---|
| `--grpo_group_size` | int | 4 | GRPO group size |
| `--epochs` | int | 0 | Number of epochs (epoch mode when >0) |
| `--steps` | int | 1500 | Maximum steps |
| `--batch_size` | int | 2 | Batch size per GPU |
| `--lr` | float | 5e-5 | Learning rate |
| `--weight_decay` | float | 0.0 | Weight decay |
| `--grad_clip` | float | 1.0 | Gradient clipping threshold |
| `--max_consecutive_nan` | int | 10 | Number of consecutive NaN/Inf losses before abort |

#### Generation Parameters

| Argument | Type | Default | Description |
|---|---|---|---|
| `--gen_max_new_tokens` | int | 192 | Maximum tokens for candidate responses |
| `--gen_temperature` | float | 0.7 | Sampling temperature |
| `--gen_top_p` | float | 0.9 | Top-p sampling |
| `--gen_repetition_penalty` | float | 1.05 | Repetition penalty |

#### Reward Mode & Parameters

Supports the same reward modes (`grpo` / `gdpo` / `gspo`) and all parameters as full-parameter GRPO. See [3. Full-Parameter GRPO Training](#reward-mode) for details.

#### KL Penalty

The LoRA version supports KL divergence penalty against the reference model (with LoRA disabled).

| Argument | Type | Default | Description |
|---|---|---|---|
| `--kl_beta` | float | 0.0 | KL penalty coefficient (0 to disable, typical: 0.01~0.1) |
| `--kl_clamp_max` | float | 10.0 | Per-sample KL upper bound |
| `--kl_x_clip` | float | 20.0 | Log-ratio clamp range for KL estimator |
| `--kl_clamp_min0` / `--no_kl_clamp_min0` | flag | True | Clamp KL to minimum 0 (GSPO only) |

#### vLLM Integration (LoRA-compatible)

The LoRA version uses vLLM's runtime LoRA load/unload API.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--use_vllm` | flag | - | Enable vLLM integration |
| `--vllm_base_url` | str | None | vLLM server URL (single) |
| `--vllm_base_urls` | str | None | Comma-separated multiple server URLs |
| `--vllm_api_key` | str | None | vLLM authentication API key |
| `--vllm_model_name` | str | env var `VLLM_MODEL_NAME` | Base model name served by vLLM |
| `--vllm_lora_name` | str | env var `VLLM_LORA_NAME` or "train-lora" | Adapter name (used as `model` field in OpenAI API) |
| `--vllm_lora_save_root` | str | env var `VLLM_LORA_SAVE_ROOT` | Adapter save path (must be readable by vLLM server) |
| `--vllm_force_n1` | flag | - | Emulate n>1 as repeated n=1 calls |
| `--vllm_connection_timeout` | int | 240 | Connection timeout (seconds) |
| `--vllm_reload_max_retries` | int | 5 | Maximum LoRA reload retries |
| `--vllm_reload_retry_backoff` | float | 2.0 | Retry backoff base (seconds) |
| `--vllm_reload_retry_jitter` | float | 0.2 | Backoff jitter factor |

#### Medical/SMART Judge Settings

Same argument structure as full-parameter GRPO. See [3. Full-Parameter GRPO Training](#medical-judge-settings) for details.

#### Save & Logging

| Argument | Type | Default | Description |
|---|---|---|---|
| `--save_dir` | str | lora_grpo_outputs | Checkpoint save directory |
| `--wandb` | flag | - | Enable W&B logging |
| `--wandb_project` | str | lora_grpo | W&B project name |
| `--wandb_run_name` | str | None | W&B run name |
| `--wandb_entity` | str | None | W&B entity |
| `--wandb_table_rows` | int | 4 | Number of batch rows to log in W&B tables |
| `--wandb_table_text_max_chars` | int | 2000 | Maximum characters for W&B table text cells |
| `--log_every` | int | 20 | Log output interval |
| `--debug_print` | flag | - | Enable debug output |
| `--debug_barrier` | flag | - | Log wait times at pre-forward barriers (rank async diagnosis) |
| `--sync_debug` | flag | - | cuda.synchronize() at phase boundaries (accurate timing, slower) |

#### Sample Output

| Argument | Type | Default | Description |
|---|---|---|---|
| `--samples_jsonl_path` | str | samples.jsonl | Sample output path |
| `--samples_jsonl_every` | int | 1 | Record every N steps (0 to disable) |
| `--samples_jsonl_overwrite` | flag | - | Overwrite at start |
| `--samples_jsonl_text_max_chars` | int | 0 | Maximum text characters (0=unlimited) |

---

## 5. vLLM Inference Server

In the GRPO training loop, vLLM servers handle candidate response generation. The launch method differs between full-parameter and LoRA versions.

### Full-Parameter vLLM Server

A custom vLLM server supporting streaming weight synchronization from FSDP.

```bash
sbatch grpo/full/script/run_vllm_qwen3_235B_servers_multi.sh
```

- 2 nodes x 8 GPUs (tensor_parallel_size=8)
- `max_model_len=8192`
- Health check: Startup confirmed via `/health` and `/v1/models` (waits up to 50 minutes)

**Custom Modules:**

- **`vllm_client.py`** (`VLLMClient`): FSDP-compatible client
  - `check_server()`: Server health check
  - `generate()` / `chat()`: Text generation / chat
  - `init_communicator()`: NCCL/gloo communicator initialization
  - `update_model_params()`: Batch parameter update
  - `update_model_params_streaming_fsdp()`: Streaming sync of FSDP shards
  - `reset_prefix_cache()`: KV cache clear
  - `close_communicator()`: Communicator cleanup

- **`vllm_serve.py`** (`WeightSyncWorkerExtension`): FastAPI server with weight sync

| Endpoint | Method | Description |
|---|---|---|
| `/health/` | GET/POST | Health check |
| `/get_world_size/` | GET | Get vLLM world size |
| `/generate/` | POST | Text generation |
| `/chat/` | POST | Chat completion |
| `/init_communicator/` | POST | NCCL/gloo communicator initialization |
| `/update_named_param/` | POST | Update a single parameter |
| `/update_param_shard/` | POST | Update a single shard |
| `/update_param_shards/` | POST | Batch shard update via metadata |
| `/reset_prefix_cache/` | POST | Prefix cache reset |
| `/close_communicator/` | POST | Communicator shutdown |

### LoRA vLLM Server

A standard vLLM server supporting runtime LoRA adapter load/unload.

```bash
sbatch grpo/lora/script/run_vllm_qwen3_235B_lora_server.sh
```

- 4 nodes x 8 GPUs (tensor_parallel_size=8)
- `max_model_len=6000`
- `max_loras=1`, `max_lora_rank=16`
- `gpu_memory_utilization=0.88`
- `max_num_seqs=64`
- LoRA API: `/v1/load_lora_adapter`, `/v1/unload_lora_adapter`
- Coordinator: Polls every 2 seconds, waits up to 60 minutes, heartbeat every 600 seconds

---

## 6. Model Merging

Three tools for integrating trained checkpoints into the base model.

### 6.1 Expert Bake-in (FSDP DCP)

Bakes trained weights of selected experts from FSDP checkpoints saved via `accelerator.save_state()` into the base model. Runs on CPU.

```bash
sbatch merge/fsdp/slurm_qwen3_235b_expert_merge.sh
```

```bash
python merge/fsdp/merge_experts_and_bake_base.py \
    --checkpoint_dir /path/to/fsdp_checkpoint \
    --base_model /path/to/Qwen3-235B-A22B-Instruct \
    --mask_json /path/to/selected_experts.json \
    --layers 94 \
    --experts 128 \
    --out_dir /path/to/merged_model \
    --dtype bf16
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--checkpoint_dir` | str | **required** | Directory saved via accelerator.save_state() |
| `--base_model` | str | **required** | HF local path to base model |
| `--mask_json` | str | **required** | Expert selection mask JSON |
| `--layers` | int | 94 | Number of model layers |
| `--experts` | int | 128 | Number of experts per layer |
| `--out_dir` | str | **required** | Output directory |
| `--dtype` | str | bf16 | Data type for loading (`bf16` / `fp16` / `fp32`) |

**Processing Flow:**
1. Load selected expert parameters from FSDP sharded checkpoint
2. Filter trainable parameters using expert mask
3. Overwrite trained weights into base model in-place
4. Fingerprint verification
5. Save in HuggingFace format via `save_pretrained()`

### 6.2 FSDP Sharded → Full Conversion

Streams and consolidates 64-way FSDP sharded checkpoints (`model_sharded.rank*.pt`) into a single full model. Uses CPU parallel processing with 64 processes via torchrun.

```bash
sbatch merge/pytorch/slurm_qwen3_235b_expert_merge.sh
```

```bash
torchrun --nproc_per_node=64 \
    merge/pytorch/fsdp_sharded_to_full_torchrun.py \
    --checkpoint_dir /path/to/sharded_ckpt \
    --base_model /path/to/Qwen3-235B-A22B-Instruct \
    --out_dir /path/to/full_model \
    --dtype bf16 \
    --backend gloo \
    --strict_dim0_only 1 \
    --send_and_free 1
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--checkpoint_dir` | str | **required** | Sharded checkpoint directory |
| `--base_model` | str | **required** | Path to base model |
| `--out_dir` | str | **required** | Output directory |
| `--dtype` | str | bf16 | Output data type (`bf16` / `fp16` / `fp32`) |
| `--backend` | str | gloo | Communication backend (`gloo` / `nccl`) |
| `--timeout_sec` | int | 3600 | Distributed timeout (seconds) |
| `--comm_dtype` | str | auto | Communication data type (`auto` / `same` / `fp32`) |
| `--strict_dim0_only` | int | 1 | Abort on detection of non-dim0 contiguous blocks |
| `--require_all_named_parameters` | int | 0 | Require all named_parameters to exist in checkpoint |
| `--send_and_free` | int | 1 | Free shard tensors after sending to save memory |
| `--log_every` | int | 2000 | Progress log output every N shards |
| `--torch_threads` | int | 1 | Threads per process |

### 6.3 LoRA Merge

Merges trained LoRA adapters into the base model and saves as a single full model. Uses PEFT's `merge_and_unload()`.

```bash
sbatch merge/lora/slurm_merge_loras.sh
```

Set the base model path and LoRA adapter paths in the SLURM script:

```bash
BASE_MODEL="/path/to/Qwen3-235B-A22B-Instruct"
LORA_DIRS=(
    "outputs/adapters/train-lora/step_00000100"
    "outputs/adapters/train-lora/step_00000075"
)
OUTPUT_DIRS=(
    "/path/to/merged_model_step100"
    "/path/to/merged_model_step75"
)
```

**Processing Flow:**
1. Load base model in BF16 (`device_map="auto"` across 8 GPUs)
2. Load LoRA adapter via `PeftModel.from_pretrained()`
3. Merge into base model via `model.merge_and_unload()`
4. Save via `save_pretrained()`

---

## 7. SLURM Configuration

### Environment Variables

Recommended settings common to all training jobs:

```bash
# NCCL
export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=NVL                    # Use NVLink
export NCCL_IB_DISABLE=0                      # Enable InfiniBand
export NCCL_BUFFSIZE=2097152                  # Buffer size
export NCCL_SOCKET_IFNAME=enp25s0np0          # High-speed network interface
export NCCL_TIMEOUT=7200                      # NCCL timeout (seconds)
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# PyTorch
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=2
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=7200 # Distributed timeout (seconds)
```

### Node Configuration Examples

| Pipeline | Nodes | GPUs/Node | Total GPUs | Notes |
|---|---|---|---|---|
| Full-parameter GRPO | 8 | 8 | 64 | accelerate launch |
| LoRA GRPO | 8 | 8 | 64 | accelerate launch |
| SFT | 8 | 8 | 64 | accelerate launch |
| DPO | 6 | 8 | 48 | accelerate launch |
| vLLM (full) | 2 | 8 | 16 | tensor_parallel_size=8 |
| vLLM (LoRA) | 4 | 8 | 32 | tensor_parallel_size=8 |
| Merge (FSDP DCP) | 1 | 0 (CPU) | - | Runs on CPU |
| Merge (PyTorch) | 1 | 0 (CPU) | - | torchrun 64 processes |
| Merge (LoRA) | 1 | 8 | 8 | device_map=auto |

### Accelerate Launch Example

```bash
accelerate launch \
    --num_machines ${SLURM_NNODES} \
    --num_processes $((SLURM_NNODES * 8)) \
    --machine_rank ${SLURM_NODEID} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MAIN_PROCESS_PORT} \
    --mixed_precision bf16 \
    script.py [args...]
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Expert Mask JSON                                        │
│  (Select experts for training)                           │
└────────────┬─────────────────────────────────────────────┘
             ▼
┌──────────────────────────────────────────────────────────┐
│  FSDP-wrapped Model (Qwen3-235B-A22B)                    │
│  Only selected experts have requires_grad=True           │
├────────────────────┬─────────────────────────────────────┤
│  Full-Param        │  LoRA Training                      │
│  Training          │  (Adapter-only updates)              │
│  (All params)      │                                     │
└────────┬───────────┴──────────┬───────────────────────────┘
         │                      │
         ▼                      ▼
┌──────────────────────────────────────────────────────────┐
│  vLLM Inference Server (Real-time weight sync)           │
│  Generates candidate responses                           │
└────────────────────┬─────────────────────────────────────┘
                     ▼
┌──────────────────────────────────────────────────────────┐
│  LLM-as-Judge Reward Evaluation                          │
│  Medical: {0, 0.5, 1}  +  SMART: 0~10                   │
│  → Reward hacking defense → Final reward                 │
└────────────────────┬─────────────────────────────────────┘
                     ▼
┌──────────────────────────────────────────────────────────┐
│  Policy Gradient Update (GRPO / GDPO / GSPO + PPO-clip)  │
└────────────────────┬─────────────────────────────────────┘
                     ▼
┌──────────────────────────────────────────────────────────┐
│  Checkpoint Save → Merge → Deploy                        │
└──────────────────────────────────────────────────────────┘
```

## License

[MIT](LICENSE), except the files under `grpo/full/trl/`, which are modified copies of HuggingFace TRL sources and remain under the Apache License 2.0. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Acknowledgements

This program is based on results obtained from a project, JPNP25006, commissioned by the New Energy and Industrial Technology Development Organization (NEDO).
