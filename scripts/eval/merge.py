#!/usr/bin/env python3
"""
Merge LoRA/PiSSA adapters into base model and save full model weights.

Usage:
    python debug.py <checkpoint_dir1> [checkpoint_dir2] ...
    
Example:
    python debug.py outputs/train/experiment1 outputs/train/experiment2
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_adapter_to_base(checkpoint_path: str) -> bool:
    """
    Merge adapter weights into base model and save full model.
    
    Args:
        checkpoint_path: Path to checkpoint directory containing adapter files
        
    Returns:
        True if successful, False otherwise
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Check if adapter files exist
    adapter_config_path = checkpoint_path / "adapter_config.json"
    adapter_weights_path = checkpoint_path / "adapter_model.safetensors"
    
    if not adapter_config_path.exists():
        print(f"  ⚠️  No adapter_config.json found in {checkpoint_path}")
        return False
    
    if not adapter_weights_path.exists():
        print(f"  ⚠️  No adapter_model.safetensors found in {checkpoint_path}")
        return False
    
    # Load adapter config to get base model path
    with open(adapter_config_path, 'r') as f:
        adapter_config = json.load(f)
    
    base_model_path = adapter_config.get("base_model_name_or_path")
    if not base_model_path:
        print(f"  ❌ No base_model_name_or_path in adapter_config.json")
        return False
    
    print(f"  📦 Base model: {base_model_path}")
    print(f"  🔧 Loading base model...")
    
    try:
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",  # Load to CPU to avoid OOM
            trust_remote_code=True,
        )
        
        print(f"  🔗 Loading adapter from {checkpoint_path.name}...")
        
        # Load adapter
        model = PeftModel.from_pretrained(
            base_model,
            str(checkpoint_path),
            device_map="cuda",
        )
        
        print(f"  🔀 Merging adapter into base model...")
        
        # Merge adapter weights into base model
        merged_model = model.merge_and_unload()
        
        print(f"  💾 Saving merged model...")
        
        # Save merged model to the same checkpoint directory
        merged_model.save_pretrained(
            str(checkpoint_path),
            safe_serialization=True,  # Save as safetensors
            max_shard_size="128GB",  # Shard if model is large
        )
        
        # Copy tokenizer files if they don't exist or update them
        if (checkpoint_path / "tokenizer.json").exists():
            print(f"  ✓ Tokenizer files already exist")
        else:
            print(f"  📝 Copying tokenizer files...")
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                trust_remote_code=True,
            )
            tokenizer.save_pretrained(str(checkpoint_path))
        
        # Copy config.json
        print(f"  📝 Copying config.json...")
        merged_model.config.save_pretrained(str(checkpoint_path))
        
        print(f"  ✅ Successfully merged and saved to {checkpoint_path}")
        
        # Clean up memory
        del model
        del merged_model
        del base_model
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error merging adapter: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_experiment_directory(exp_dir: str) -> None:
    """
    Process all checkpoint-* directories in an experiment directory.
    
    Args:
        exp_dir: Path to experiment directory containing checkpoint-* subdirs
    """
    exp_path = Path(exp_dir)
    
    if not exp_path.exists():
        print(f"❌ Directory not found: {exp_dir}")
        return
    
    # Find all checkpoint directories
    checkpoint_dirs = sorted(exp_path.glob("checkpoint-*"))
    
    if not checkpoint_dirs:
        print(f"⚠️  No checkpoint-* directories found in {exp_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing experiment: {exp_path.name}")
    print(f"Found {len(checkpoint_dirs)} checkpoint(s)")
    print(f"{'='*60}\n")
    
    success_count = 0
    fail_count = 0
    
    for checkpoint_dir in checkpoint_dirs:
        print(f"\n📁 Processing {checkpoint_dir.name}...")
        
        # Check if already merged (model.safetensors exists)
        if (checkpoint_dir / "model.safetensors").exists() or \
           (checkpoint_dir / "model.safetensors.index.json").exists():
            print(f"  ⏭️  Model already merged (model.safetensors found), skipping...")
            continue
        
        if merge_adapter_to_base(str(checkpoint_dir)):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n{'='*60}")
    print(f"Summary for {exp_path.name}:")
    print(f"  ✅ Successfully merged: {success_count}")
    print(f"  ❌ Failed: {fail_count}")
    print(f"{'='*60}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python debug.py <checkpoint_dir1> [checkpoint_dir2] ...")
        print("\nExample:")
        print("  python debug.py outputs/train/experiment1")
        print("  python debug.py outputs/train/exp1 outputs/train/exp2")
        sys.exit(1)
    
    experiment_dirs = sys.argv[1:]
    
    print(f"\n🚀 Starting adapter merge process...")
    print(f"📂 Processing {len(experiment_dirs)} experiment directory(ies)\n")
    
    for exp_dir in experiment_dirs:
        process_experiment_directory(exp_dir)
    
    print("\n✨ All done!")


if __name__ == "__main__":
    main()
