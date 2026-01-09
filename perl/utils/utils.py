import sys
import ast

from perl.config.config import TrainConfig, SFTTrainConfig

# System prompt
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def _parse_args_to_config(config):
    """Generic config parser - parses --config.* arguments into a config object."""
    # Parse --config.* arguments, skip the subcommand (first arg after script name)
    args = sys.argv[2:]  # Skip script name and subcommand
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith('--config.'):
            config_path = arg[len('--config.'):]
            if i + 1 < len(args):
                value_str = args[i + 1]
                # Try to parse the value
                try:
                    value = int(value_str)
                except ValueError:
                    try:
                        value = float(value_str)
                    except ValueError:
                        if value_str.lower() in ('true', 'false'):
                            value = value_str.lower() == 'true'
                        else:
                            if value_str.startswith('[') and value_str.endswith(']'):
                                value = ast.literal_eval(value_str)
                            else:
                                value = value_str

                # Set the nested attribute
                parts = config_path.split('.')
                obj = config
                for part in parts[:-1]:
                    if isinstance(obj, dict):
                        if part not in obj:
                            obj[part] = {}
                        obj = obj[part]
                    else:
                        if not hasattr(obj, part):
                            raise ValueError(f"Unknown config section: {part}")
                        obj = getattr(obj, part)
                
                attr_name = parts[-1]
                if isinstance(obj, dict):
                    obj[attr_name] = value
                else:
                    if not hasattr(obj, attr_name):
                        raise ValueError(f"Unknown config attribute: {attr_name}")
                    setattr(obj, attr_name, value)
                i += 2
            else:
                raise ValueError(f"Missing value for {arg}")
        else:
            i += 1

    return config


def parse_grpo_config():
    """Parse command line arguments for GRPO training."""
    return _parse_args_to_config(TrainConfig())


def parse_sft_config():
    """Parse command line arguments for SFT training."""
    return _parse_args_to_config(SFTTrainConfig())


# Backward compatibility alias
parse_args_to_config = parse_grpo_config
