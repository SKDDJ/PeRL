# main.py
import argparse
import asyncio
import atexit
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Set path to include current directory for imports
os.environ["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from eval.utils import setup_logging, StageContext, merge_model_if_needed
from eval.vllm import (
    start_vllm_processes, 
    stop_vllm_processes, 
    wait_for_vllm_ready, 
    generate_responses, 
    evaluate_dataset_results
)

def parse_args() -> Tuple[argparse.Namespace, List[str], List[str]]:
    parser = argparse.ArgumentParser(
        description="Evaluation entry script, supports model merging, vLLM startup, and multi-dataset evaluation."
    )
    parser.add_argument("--result-dir", required=True, help="Directory for intermediate processes and result output.")
    parser.add_argument("--model", required=True, help="Base model name or path.")
    parser.add_argument(
        "--adapter", default="", help="LoRA/PEFT adapter path, leave empty for no merge."
    )
    parser.add_argument(
        "--dataset",
        default="aime2024",
        help="Dataset abbreviation to evaluate, comma separated (e.g., aime2024).",
    )
    parser.add_argument(
        "--rollout-n", type=int, default=1, help="Number of rollouts to generate per sample."
    )
    parser.add_argument(
        "--serve-port", type=int, default=8000, help="First vLLM backend port number."
    )
    parser.add_argument(
        "--dp-size", type=int, default=1, help="Number of data parallel backends (start multiple vLLMs)."
    )
    parser.add_argument(
        "--tp-size", type=int, default=1, help="Tensor parallel size passed to vLLM."
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="Verify needed GPU count before running, error if insufficient."
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
        help="GPU memory utilization limit passed to vLLM (0~1), controls memory usage per card.",
    )
    parser.add_argument("--seed", type=float, default=None, help="Generation random seed.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Generation temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Generation top-p.")
    parser.add_argument("--max-new-tokens", type=int, default=131072, help="Generation length.")
    parser.add_argument("--dtype", default="auto", help="Model dtype, used during merging.")
    parser.add_argument(
        "--trust-remote-code", action="store_true", help="Whether to trust remote code."
    )
    parser.add_argument(
        "--served-model-name", default="eval-model", help="Model name exposed by vLLM."
    )
    parser.add_argument("--api-key", default="dummy", help="API Key for OpenAI compatible interface.")
    parser.add_argument(
        "--request-timeout", type=float, default=3600.0, help="Timeout for a single request."
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="For debugging, limit number of evaluation samples."
    )
    parser.add_argument(
        "--max-num-request",
        type=int,
        default=None,
        help="Max number of concurrent requests per data parallel (DP) vLLM backend.",
    )

    args, unknown = parser.parse_known_args()

    if args.max_num_request is None:
        args.max_num_request = args.dp_size
    else:
        assert args.max_num_request > 0
        assert args.max_num_request % args.dp_size == 0, (
            f"args.max_num_request({args.max_num_request}) must be divisible by args.dp_size({args.dp_size})"
        )

    vllm_args, leftover = extract_vllm_args(unknown)
    return args, vllm_args, leftover

def extract_vllm_args(unknown: List[str]) -> Tuple[List[str], List[str]]:
    vllm_args: List[str] = []
    leftover: List[str] = []
    idx = 0
    while idx < len(unknown):
        token = unknown[idx]
        if token.startswith("--vllm-"):
            stripped = "--" + token[len("--vllm-") :]
            if "=" in token:
                _, value = token.split("=", 1)
                vllm_args.extend([stripped, value])
            elif idx + 1 < len(unknown) and not unknown[idx + 1].startswith("-"):
                vllm_args.extend([stripped, unknown[idx + 1]])
                idx += 1
            else:
                vllm_args.append(stripped)
        else:
            leftover.append(token)
        idx += 1
    return vllm_args, leftover

async def main() -> None:
    args, vllm_args, leftover = parse_args()
    logger = setup_logging(Path(args.result_dir))
    if leftover:
        logger.warning("Detected unrecognized arguments (will be ignored): %s", leftover)

    with StageContext(logger, "A", "Prepare Model/Merge LoRA"):
        model_path = merge_model_if_needed(args, Path(args.result_dir), logger)

    with StageContext(logger, "B", "Start vLLM Backends"):
        processes, ports = start_vllm_processes(model_path, args, vllm_args, logger)
        atexit.register(stop_vllm_processes, processes, logger)

        def handle_signal(signum, frame):  # noqa: ANN001
            logger.warning("Received signal %d, preparing to clean up and exit.", signum)
            stop_vllm_processes(processes, logger)
            sys.exit(1)

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        for proc, port in zip(processes, ports):
            if not wait_for_vllm_ready(port, proc, timeout=300, logger=logger):
                stop_vllm_processes(processes, logger)
                sys.exit(1)

    # Initialize global semaphores
    dp_size = max(1, args.dp_size)
    max_concurrent_per_dp = max(1, args.max_num_request // dp_size)
    semaphores = {port: asyncio.Semaphore(max_concurrent_per_dp) for port in ports}
    logger.info("Global concurrency control initialized: Max concurrency per DP process=%d", max_concurrent_per_dp)

    async def process_dataset_task(
        args: argparse.Namespace,
        dataset_name: str,
        rollout_n: int,
        ports: List[int],
        logger: logging.Logger,
        semaphores: Dict[int, asyncio.Semaphore],
    ) -> None:
        with StageContext(logger, f"C[{dataset_name}]", "Dataset Generation (Cache/Gen)"):
            await generate_responses(
                args, dataset_name, rollout_n, ports, logger, semaphores
            )

        with StageContext(logger, f"D[{dataset_name}]", "Evaluation & Statistics"):
            # evaluate_dataset_results is synchronous CPU-bound task, put in thread pool to avoid blocking other concurrent tasks
            await asyncio.to_thread(
                evaluate_dataset_results, args, dataset_name, rollout_n, logger
            )

    datasets_to_run = [item.strip() for item in args.dataset.split(",") if item.strip()]
    tasks = []

    for task_abbr in datasets_to_run:
        if "@" in task_abbr:
            dataset_name = task_abbr.split("@")[0]
            rollout_n = int(task_abbr.split("@")[1])
        else:
            dataset_name = task_abbr
            rollout_n = args.rollout_n

        tasks.append(
            asyncio.create_task(
                process_dataset_task(
                    args, dataset_name, rollout_n, ports, logger, semaphores
                )
            )
        )

    if tasks:
        logger.info("Submitted %d dataset tasks concurrently, starting execution...", len(tasks))
        await asyncio.gather(*tasks)
    else:
        logger.warning("No dataset tasks to execute.")

    stop_vllm_processes(processes, logger)
    logger.info("All evaluation processes completed.")

if __name__ == "__main__":
    asyncio.run(main())