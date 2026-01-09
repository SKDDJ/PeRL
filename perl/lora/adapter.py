# copyright (c) 2025, mikastars39.org
# All rights reserved.
# This source code is licensed under the Apache-2.0 License.
# See the LICENSE file in the root directory for details.

from .slicefine import register_slicefine_method
register_slicefine_method() # register slicefine method to peft
from perl.utils.logging import logger

def apply_lora(model, args):
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(
        peft_type="LORA",
        task_type=args.peft.task_type,
        r=args.peft.r,
        lora_alpha=args.peft.lora_alpha,
        target_modules=args.peft.target_modules,
        lora_dropout=args.peft.lora_dropout,
    )
    return None, get_peft_model(model, config)

def apply_dora(model, args):
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(
        peft_type="LORA",
        use_dora=True,
        task_type=args.peft.task_type,
        r=args.peft.r,
        lora_alpha=args.peft.lora_alpha,
        target_modules=args.peft.target_modules,
        lora_dropout=args.peft.lora_dropout,
    )
    return None, get_peft_model(model, config)

def apply_vera(model, args):
    from peft import VeraConfig, get_peft_model
    config = VeraConfig(r=args.peft.r)
    return None, get_peft_model(model, config)

def apply_miss(model, args):
    from peft import MissConfig, get_peft_model
    config = MissConfig(r=args.peft.r)
    return None, get_peft_model(model, config)

def apply_pissa(model, args):
    """
    Apply PiSSA (Principal Singular values and Singular vectors Adaptation) to the model.
    
    PiSSA decomposes original weights W into principal components (A_0 * B_0) and residual (W_residual).
    The backbone is replaced with W_residual, and LoRA A/B are initialized with A_0, B_0.
    
    CRITICAL: When saving PiSSA, we must convert it to standard LoRA format for correct evaluation.
    ΔW = A*B - A_0*B_0 = [A | A_0] * [B | -B_0]^T = A' * B'
    
    This conversion is done automatically by passing path_initial_model_for_weight_conversion
    to save_pretrained(). We monkey-patch the method to ensure all saves (including intermediate
    checkpoints) are properly converted.
    """
    import os
    from peft import LoraConfig, get_peft_model
    
    lora_config = LoraConfig(
        # init_lora_weights="pissa", # Configure the initialization method to "pissa", which may take several minutes to execute SVD on the pre-trained model.
        init_lora_weights="pissa_niter_4", # Initialize the PiSSA with fast SVD, which completes in just a few seconds.
        r=args.peft.r,
        lora_alpha=args.peft.lora_alpha,
        lora_dropout=args.peft.lora_dropout, # Since the component of the PiSSA adapter are the principal singular values and vectors, dropout should be set to 0 to avoid random discarding.
        target_modules=args.peft.target_modules,
        task_type=args.peft.task_type,
    )
    peft_model = get_peft_model(model, lora_config)
    
    # Save PiSSA initial weights for later conversion
    # This directory contains A_0, B_0 which are needed to convert PiSSA -> LoRA
    pissa_init_dir = os.path.join(args.training.output_dir, "pissa_init")
    os.makedirs(pissa_init_dir, exist_ok=True)
    peft_model.save_pretrained(pissa_init_dir)
    print(f"[PiSSA] Saved initial weights to {pissa_init_dir} for PiSSA -> LoRA conversion")
    
    # Monkey-patch save_pretrained to automatically convert PiSSA to standard LoRA format
    # This ensures all saves (intermediate checkpoints + final save) are correctly converted
    original_save_pretrained = peft_model.save_pretrained
    
    def save_pretrained_with_pissa_conversion(save_directory, *save_args, **save_kwargs):
        """
        Wrapper around save_pretrained that automatically converts PiSSA to LoRA format.
        
        By passing path_initial_model_for_weight_conversion, PEFT computes:
        ΔW = A*B - A_0*B_0 and saves this as standard LoRA weights that can be loaded
        on the original (non-decomposed) base model.
        """
        if 'path_initial_model_for_weight_conversion' not in save_kwargs:
            save_kwargs['path_initial_model_for_weight_conversion'] = pissa_init_dir
        
        if 'state_dict' in save_kwargs:
            # Fix for KeyError: 'model.model.layers...' mismatch
            # The state_dict passed by Trainer has 'base_model.' prefix (because it comes from PeftModel),
            # whereas `subtract_mutated_init` (invoked on the underlying LoraModel) expects keys 
            # relative to the LoraModel itself.
            # We explicitly strip the 'base_model.' prefix to align the keys.
            original_state_dict = save_kwargs['state_dict']
            new_state_dict = {}
            for k, v in original_state_dict.items():
                if k.startswith("base_model."):
                    new_state_dict[k.replace("base_model.", "", 1)] = v
                else:
                    new_state_dict[k] = v
            save_kwargs['state_dict'] = new_state_dict


        return original_save_pretrained(save_directory, *save_args, **save_kwargs)
    
    peft_model.save_pretrained = save_pretrained_with_pissa_conversion
    
    return None, peft_model

def apply_milora(model, args):
    from .milora import add_svd_initialized_lora
    return None, add_svd_initialized_lora(
        model=model,
        rank=args.peft.r,
    )

def apply_layernorm(model, args):
    from peft import get_peft_model, TaskType, LNTuningConfig
    peft_config = LNTuningConfig(
        task_type=TaskType.CAUSAL_LM,
    )
    return None, get_peft_model(model, peft_config)

def apply_adalora(model, args):
    from peft import AdaLoraConfig, get_peft_model
    config = AdaLoraConfig(
        peft_type="ADALORA",
        task_type=args.peft.task_type,
        init_r=args.peft.r,
        lora_alpha=args.peft.lora_alpha,
        target_modules=args.peft.target_modules,
        lora_dropout=args.peft.lora_dropout,
        total_step=args.training.max_steps,
    )
    return None, get_peft_model(model, config)

def apply_IA3(model, args):
    from peft import IA3Config, get_peft_model, TaskType
    config = IA3Config(task_type=TaskType.CAUSAL_LM)
    return None, get_peft_model(model, config)

def apply_milora_plus(model, args):
    from .milora_plus import add_svd_initialized_lora
    return None, add_svd_initialized_lora(
        model=model,
        rank=args.peft.r,
    )

def apply_lorafa(model, args):
    from peft import LoraConfig, get_peft_model
    from peft.optimizers import create_lorafa_optimizer

    config = LoraConfig(
        peft_type="LORA",
        task_type=args.peft.task_type,
        r=args.peft.r,
        lora_alpha=args.peft.lora_alpha,
        target_modules=args.peft.target_modules,
        lora_dropout=args.peft.lora_dropout,
    ) 

    model = get_peft_model(model, config)
        
    optimizer = create_lorafa_optimizer(
        model=model,
        r=args.peft.r,
        lora_alpha=args.peft.lora_alpha,
        lr=args.training.learning_rate,
    )
    return optimizer, model

def apply_lora_plus(model, args):
    from peft import LoraConfig, get_peft_model
    from torch.optim import AdamW
    from peft.optimizers import create_loraplus_optimizer
    
    # First create the LoRA model
    config = LoraConfig(
        peft_type="LORA",
        task_type=args.peft.task_type,
        r=args.peft.r,
        lora_alpha=args.peft.lora_alpha,
        target_modules=args.peft.target_modules,
        lora_dropout=args.peft.lora_dropout,
    )
    model = get_peft_model(model, config)
    
    # Then create the LoraPlus optimizer with different learning rates for A and B
    optimizer = create_loraplus_optimizer(
        model=model,
        optimizer_cls=AdamW,
        lr=args.training.learning_rate,
        loraplus_lr_ratio=2.0,
    )
    return optimizer, model

def apply_slicefine(model, args):
    from .slicefine import SliceFineConfig
    from peft import get_peft_model
    config = SliceFineConfig(
        r=args.peft.r,
        slice_mode=getattr(args.peft, "slice_mode", "column"),
        slice_position=getattr(args.peft, "slice_position", 0),
        target_modules=args.peft.target_modules,
        bias="all" if getattr(args.peft, "bias", False) else "none"
    )
    print(f"[SliceFine] Applying SliceFine with rank={config.r}, modules={config.target_modules}")
    
    peft_model = get_peft_model(model, config)
    
    peft_model.print_trainable_parameters()
    
    trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError(
            "[SliceFine Error] No trainable parameters found! \n"
            "1. Check if 'target_modules' match the model architecture.\n"
            "2. Check if 'part_T' is correctly set to requires_grad=True."
        )
    return None, peft_model

def apply_hra(model, args):
    from peft import HRAConfig, get_peft_model
    config = HRAConfig(
        r=args.peft.r,
        target_modules=args.peft.target_modules,
        init_weights="True",
    )
    return None, get_peft_model(model, config)

def apply_rslora(model, args):
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(
        peft_type="LORA",
        use_rslora=True,
        task_type=args.peft.task_type,
        r=args.peft.r,
        lora_alpha=args.peft.lora_alpha,
        target_modules=args.peft.target_modules,
        lora_dropout=args.peft.lora_dropout,
    )
    return None, get_peft_model(model, config)

# ------------------------------ mapping function to peft type ------------------------------

PEFT_TYPE_TO_FUNCTION_MAPPING = {
    "lora": apply_lora,
    "dora": apply_dora,
    "slicefine": apply_slicefine,
    "milora": apply_milora,
    "milora_plus": apply_milora_plus,
    "lorafa": apply_lorafa,
    "lora_plus": apply_lora_plus,
    "adalora": apply_adalora,
    "IA3": apply_IA3,
    "layernorm": apply_layernorm,
    "vera": apply_vera,
    "miss": apply_miss,
    "pissa": apply_pissa,
    "hra": apply_hra,
}

# ------------------------------ dispatch function to peft type ------------------------------

def apply_peft(model, args):
    """Dispatch function that routes to the appropriate PEFT method based on args.peft.type"""
    peft_type = args.peft.type
    if peft_type in PEFT_TYPE_TO_FUNCTION_MAPPING:
        return PEFT_TYPE_TO_FUNCTION_MAPPING[peft_type](model, args)
    else:
        raise ValueError(f"Unsupported PEFT type: {peft_type}")