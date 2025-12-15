from .lora.adapter import apply_peft, PEFT_TYPE_TO_FUNCTION_MAPPING
from transformers import AutoModelForCausalLM

from argparse import Namespace

def param_test(
    r: int = 32,
):
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    results = {}
     
    for peft_type in PEFT_TYPE_TO_FUNCTION_MAPPING.keys():
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        args = Namespace(
            peft=Namespace(
                type=peft_type,
                r=r,
                lora_alpha=64,
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj"],
            ),
            training=Namespace(
                learning_rate=1e-5,
                beta=0.0,
                max_steps=1000,
            )
        )
        
        try:
            optimizer, peft_model = PEFT_TYPE_TO_FUNCTION_MAPPING[peft_type](model, args)
            trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
            trainable_params_m = trainable_params / 1_000_000
            results[peft_type] = trainable_params_m
            print(f"{peft_type}: {trainable_params_m:.4f}M")
        except Exception as e:
            print(f"{peft_type}: Error - {str(e)}")
            results[peft_type] = None
             
    return results

if __name__ == "__main__":
    results = param_test()