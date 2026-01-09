# Copyright (c) 2025, PeRL Authors
# SFT (Supervised Fine-Tuning) training module

import torch
import os

from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

from perl.utils.logging import init_logger, logger
from perl.data import load_sft_dataset
from perl.config.config import SFTTrainConfig


def setup_sft_environment(args: SFTTrainConfig):
    """Setup environment for SFT training."""
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    is_main_process = local_rank == 0
    
    init_logger(rank=local_rank)
    args.training.output_dir = args.training.output_dir or "output_sft"
    args.training.run_name = args.training.run_name or args.training.output_dir
    
    if is_main_process:
        if not os.path.exists(args.training.output_dir):
            os.makedirs(args.training.output_dir, exist_ok=True)
        else:
            logger.info(f"Output directory {args.training.output_dir} already exists")
    set_seed(args.common.seed)

    if args.common.debug:
        args.training.report_to = []

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    if is_main_process:
        if "wandb" in args.training.report_to:
            import wandb
            wandb.init(
                project=args.logging.wandb_project,
                name=args.training.run_name,
                config=vars(args.training),
            )
            logger.info("Wandb initialized successfully")

    return args, is_main_process


def sft(config: SFTTrainConfig = None):
    """SFT training entry point."""
    print(config)
    args, is_main_process = setup_sft_environment(config)

    # 1. Load tokenizer
    logger.info(f"Loading tokenizer from {args.model.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 2. Load dataset
    logger.info(f"Loading SFT dataset from {args.dataset.dataset_name_or_path}")
    dataset = load_sft_dataset(
        args.dataset.dataset_name_or_path,
        example_numbers=args.dataset.example_numbers,
        tokenizer=tokenizer
    )
    train_dataset = dataset["train"]
    eval_dataset = dataset.get("test", None)

    # 3. Load model
    logger.info(f"Loading model from {args.model.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.model.dtype == "bfloat16" else torch.float16,
        attn_implementation="flash_attention_2"
    )
    logger.info("Model loaded successfully")

    # 4. Configure PEFT if enabled
    optimizer = None
    if args.peft.use_peft:
        logger.info(f"Configuring PEFT: {args.peft.type}")
        from perl.lora.adapter import apply_peft
        optimizer, model = apply_peft(model, args)
        model.print_trainable_parameters()
        logger.info("PEFT configured successfully")

    # 5. Training configuration
    training_args = SFTConfig(
        output_dir=args.training.output_dir,
        run_name=args.training.run_name,
        learning_rate=args.training.learning_rate,
        per_device_train_batch_size=args.training.per_device_train_batch_size,
        gradient_accumulation_steps=args.training.gradient_accumulation_steps,
        num_train_epochs=args.training.num_train_epochs,
        max_steps=args.training.max_steps,
        max_length=args.training.max_seq_length,
        packing=args.training.packing,
        logging_steps=args.training.logging_steps,
        save_strategy=args.training.save_strategy,
        save_steps=args.training.save_steps,
        lr_scheduler_type=args.training.lr_scheduler_type,
        warmup_ratio=args.training.warmup_ratio,
        bf16=args.training.bf16,
        gradient_checkpointing=args.training.gradient_checkpointing,
        report_to=args.training.report_to,
    )

    # 6. Train
    logger.info("Starting SFT training")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        optimizers=(optimizer, None) if optimizer is not None else (None, None),
    )
    
    resume_checkpoint = args.training.resume_from_checkpoint
    if resume_checkpoint == "true":
        resume_checkpoint = True
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    logger.info("SFT training completed successfully")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
