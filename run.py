import sys

def main():
    """
    PeRL Training Entry Point
    
    Usage:
        python run.py grpo --config.xxx ...   # GRPO/RL training
        python run.py sft --config.xxx ...    # SFT training
    """
    if len(sys.argv) < 2:
        print("Usage: python run.py <command> [options]")
        print("Commands:")
        print("  grpo    GRPO/RL training")
        print("  sft     SFT (Supervised Fine-Tuning)")
        print("  train   Alias for grpo (deprecated)")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command in ["grpo", "train"]:
        from perl.grpo import grpo
        from perl.utils.utils import parse_grpo_config
        config = parse_grpo_config()
        grpo(config)
    elif command == "sft":
        from perl.sft import sft
        from perl.utils.utils import parse_sft_config
        config = parse_sft_config()
        sft(config)
    else:
        print(f"Unknown command: {command}")
        print("Available commands: grpo, sft, train")
        sys.exit(1)


if __name__ == "__main__":
    main()