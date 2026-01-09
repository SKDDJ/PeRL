<div align="center">

# PeRL: Parameter-Efficient Reinforcement Learning  
> A minimal, modular, and lightning-fast framework for PEFT + RL/SFT training.
</div>

## 🧩 Supported Parameter-Efficient Methods

| Method | Status | Description |
| :--- | :--- | :--- |
| **LoRA** | ✅ | Standard Low-Rank Adaptation |
| **DoRA** | ✅ | Weight-decomposed Low-Rank Adaptation |
| **MiSS** | ✅ | Mixture of Sub-Spaces (Efficient shard-sharing structure) |
| **AdaLoRA** | ✅ | Adaptive budget allocation for rank-adaptive matrices |
| **LoRA+** | ✅ | Differentiated learning rates for improved adaptation dynamics |
| **rsLORA** | ✅ | Rank stabilization scaling factors |
| **PiSSA** | ✅ | Principal Singular values & Singular vectors Adaptation |
| **MiLORA** | ✅ | Minor Singular components initialization |
| **LORA-FA** | ✅ | Memory-efficient adaptation with frozen projection matrix A |
| **VeRA** | ✅ | Vector-based Random Matrix Adaptation |
| **LN Tuning** | ✅ | Parameter-efficient tuning on Layer Normalization layers |
| **$IA^3$** | ✅ | Infused Adapter by Inhibiting and Amplifying Inner Activations |

## 🚀 Features

- **SFT Training** - Supervised Fine-Tuning with TRL's SFTTrainer
- **GRPO Training** - Group Relative Policy Optimization (RL)
- **SFT → RL Pipeline** - Seamless two-stage training via checkpoint inheritance
- **Multi-GPU Support** - DeepSpeed ZeRO-2/3 with Accelerate

## Environment Setup

```bash
uv pip install -r requirements.txt
uv pip install flash-attn --no-cache-dir --no-build-isolation
python -c "import flash_attn" # verify
```

## Training

### SFT Training

```bash
python run.py sft \
    --config.model.model_name_or_path "Qwen/Qwen2.5-1.5B-Instruct" \
    --config.peft.type "lora" \
    --config.training.output_dir outputs/sft_model \
    --config.dataset.dataset_name_or_path "YOUR_SFT_DATASET"
```

### GRPO/RL Training

```bash
python run.py grpo \
    --config.model.model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --config.peft.type "pissa" \
    --config.training.output_dir outputs/grpo_model \
    --config.dataset.dataset_name_or_path "open-r1/DAPO-Math-17k-Processed"
```

### SFT → RL Pipeline

```bash
# Step 1: SFT
python run.py sft --config.training.output_dir outputs/sft_stage1 ...

# Step 2: GRPO (load SFT checkpoint)
python run.py grpo --config.model.model_name_or_path outputs/sft_stage1 ...
```

### Example Scripts

```bash
bash scripts/sft/example_sft.sh      # SFT training
bash scripts/openr1/dapo_lora.sh     # GRPO with LoRA
bash scripts/openr1/dapo_pissa.sh    # GRPO with PiSSA
```