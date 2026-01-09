from .openr1 import load_openr1_dataset
from .tinyzero import load_tinyzero_dataset
from .count_down import load_count_down_dataset
from .still import load_still_dataset

from transformers import AutoTokenizer
from datasets import load_dataset as hf_load_dataset


def load_dataset(dataset_name_or_path: str, example_numbers: int = None, tokenizer: AutoTokenizer = None):
    """Load dataset for GRPO/RL training (returns reward_functions)."""
    dataset_name_lower = dataset_name_or_path.lower()
    if "r1" in dataset_name_lower:
        return load_openr1_dataset(dataset_name_or_path, example_numbers)
    elif "tinyzero" in dataset_name_lower:
        return load_tinyzero_dataset(dataset_name_or_path, example_numbers)
    elif "countdown" in dataset_name_lower:
        return load_count_down_dataset(
            dataset_name_or_path, 
            example_numbers, 
            tokenizer
        )
    elif "still" in dataset_name_lower:
        return load_still_dataset(dataset_name_or_path, example_numbers)
    else:
        raise ValueError(f"Not supported dataset: {dataset_name_or_path}")


def load_sft_dataset(dataset_name_or_path: str, example_numbers: int = None, tokenizer: AutoTokenizer = None):
    """
    Load dataset for SFT training.
    
    Expects dataset with 'messages' column in standard chat format:
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    
    Or dataset with 'prompt' and 'response'/'completion' columns that will be converted.
    
    Args:
        dataset_name_or_path: HuggingFace dataset name or local path
        example_numbers: Maximum number of examples to use
        tokenizer: Tokenizer for chat template application
    
    Returns:
        dict with 'train' and optionally 'test' splits
    """
    # Load from HuggingFace Hub or local path
    dataset = hf_load_dataset(dataset_name_or_path)
    
    # Limit examples if specified
    if example_numbers is not None and example_numbers > 0:
        for split in dataset:
            if len(dataset[split]) > example_numbers:
                dataset[split] = dataset[split].select(range(example_numbers))
    
    # Check if dataset already has 'messages' column (standard chat format)
    sample = dataset["train"][0] if "train" in dataset else list(dataset.values())[0][0]
    
    if "messages" in sample:
        # Already in chat format, return as-is
        return dataset
    
    # Try to convert from prompt/response format to messages format
    def convert_to_messages(example):
        """Convert prompt/response to standard chat messages format."""
        messages = []
        
        # Handle various column naming conventions
        if "prompt" in example:
            messages.append({"role": "user", "content": example["prompt"]})
        elif "instruction" in example:
            content = example["instruction"]
            if "input" in example and example["input"]:
                content = f"{content}\n\n{example['input']}"
            messages.append({"role": "user", "content": content})
        elif "question" in example:
            messages.append({"role": "user", "content": example["question"]})
        
        # Handle response/completion/output
        if "response" in example:
            messages.append({"role": "assistant", "content": example["response"]})
        elif "completion" in example:
            messages.append({"role": "assistant", "content": example["completion"]})
        elif "output" in example:
            messages.append({"role": "assistant", "content": example["output"]})
        elif "answer" in example:
            messages.append({"role": "assistant", "content": example["answer"]})
        
        return {"messages": messages}
    
    # Apply conversion
    dataset = dataset.map(convert_to_messages, remove_columns=dataset["train"].column_names if "train" in dataset else list(dataset.values())[0].column_names)
    
    return dataset