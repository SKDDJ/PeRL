
import argparse
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import asyncio
import json
import statistics
import aiohttp
from typing import List, Tuple, Iterable, Set, Any, Dict
from pathlib import Path

from .utils import load_dataset_from_hf, prepare_prompt, score_response, StageContext

def build_vllm_command(
    model_path: Path, port: int, args: argparse.Namespace, vllm_args: List[str]
) -> List[str]:
    dp_size = max(1, args.dp_size)
    max_concurrent_per_dp = max(1, args.max_num_request // dp_size)

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model_path),
        "--served-model-name",
        args.served_model_name,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(args.tp_size),
        "--max-num-seqs",
        str(max_concurrent_per_dp),
    ]

    # Solution: append --gpu-memory-utilization to vLLM command, default 0.95, configurable via CLI.
    if args.gpu_memory_utilization is not None:
        cmd.extend(["--gpu-memory-utilization", str(args.gpu_memory_utilization)])
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    cmd.extend(vllm_args)
    return cmd

def pipe_to_logger(
    stream: Iterable[str], logger: logging.Logger, level: int, prefix: str
) -> None:
    for line in stream:
        logger.log(level, "%s%s", prefix, line.rstrip("\n"))

def start_vllm_processes(
    model_path: Path,
    args: argparse.Namespace,
    vllm_args: List[str],
    logger: logging.Logger,
) -> Tuple[List[subprocess.Popen], List[int]]:
    ports: List[int] = []
    processes: List[subprocess.Popen] = []
    env = os.environ.copy()
    dp_size = max(1, args.dp_size)

    for rank in range(dp_size):
        # Calculate GPU ID range for current process
        start_gpu_id = rank * args.tp_size
        end_gpu_id = start_gpu_id + args.tp_size
        gpu_ids = list(range(start_gpu_id, end_gpu_id))

        # Check for out-of-bounds (based on args.num_gpus or simple logic, assuming user config is correct)
        
        env_local = env.copy()
        env_local["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

        port = args.serve_port + rank
        cmd = build_vllm_command(model_path, port, args, vllm_args)
        logger.info(
            "Starting vLLM backend [%d/%d], port %d, GPUs=%s, command: %s",
            rank + 1,
            dp_size,
            port,
            gpu_ids,
            " ".join(cmd),
        )
        proc = subprocess.Popen(
            cmd,
            env=env_local,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )
        processes.append(proc)
        ports.append(port)
        if proc.stdout:
            threading.Thread(
                target=pipe_to_logger,
                args=(proc.stdout, logger, logging.INFO, f"[vllm:{port}] "),
                daemon=True,
            ).start()
        if proc.stderr:
            threading.Thread(
                target=pipe_to_logger,
                args=(proc.stderr, logger, logging.ERROR, f"[vllm:{port}] "),
                daemon=True,
            ).start()
    return processes, ports

def stop_vllm_processes(
    processes: List[subprocess.Popen], logger: logging.Logger
) -> None:
    for proc in processes:
        if proc.poll() is None:
            try:
                logger.info("Attempting to terminate vLLM process (pid=%d).", proc.pid)
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Exception during process termination (pid=%d): %s", proc.pid, exc)
    for proc in processes:
        if proc.poll() is None:
            try:
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass

def wait_for_vllm_ready(
    port: int, process: subprocess.Popen, timeout: float, logger: logging.Logger
) -> bool:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if process.poll() is not None:
            logger.error("vLLM process (pid=%d) exited prematurely.", process.pid)
            return False
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    logger.info("vLLM at port %d is ready.", port)
                    return True
        except Exception:
            time.sleep(2)
    logger.error("Timeout waiting for vLLM at port %d.", port)
    return False

async def generate_with_vllm_async(
    session: aiohttp.ClientSession, prompt: str, port: int, args: argparse.Namespace
) -> str:
    """Async version of vLLM generation function for concurrent requests."""
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "model": args.served_model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_new_tokens,
        "n": 1,
    }
    if args.seed is not None:
        payload["seed"] = args.seed
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    try:
        async with session.post(
            url, json=payload, headers=headers, timeout=timeout
        ) as response:
            if response.status != 200:
                raise RuntimeError(f"vLLM returned HTTP error: {response.status}")
            content = await response.json()
    except aiohttp.ClientError as exc:
        raise RuntimeError(f"vLLM connection failed: {exc}") from exc

    try:
        return content["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to parse vLLM response: {content}") from exc

# -------------- engine utils ---------------------

class ProgressVisualizer:
    def __init__(
        self,
        filepath: Path,
        problem_n: int,
        rollout_n: int,
        completed: Set[Tuple[int, int]],
    ) -> None:
        self.filepath = filepath
        self.problem_n = problem_n
        self.rollout_n = rollout_n
        # Row: rollout_id, Col: problem_id
        self.grid = [["." for _ in range(problem_n)] for _ in range(rollout_n)]
        for pid, rid in completed:
            if 0 <= rid < rollout_n and 0 <= pid < problem_n:
                self.grid[rid][pid] = "X"
        self.lock = asyncio.Lock()
        self._write_sync()

    def _write_sync(self) -> None:
        try:
            with self.filepath.open("w", encoding="utf-8") as f:
                for row in self.grid:
                    f.write("".join(row) + "\n")
        except Exception:
            pass

    async def update(self, problem_id: int, rollout_id: int) -> None:
        if 0 <= rollout_id < self.rollout_n and 0 <= problem_id < self.problem_n:
            async with self.lock:
                if self.grid[rollout_id][problem_id] != "X":
                    self.grid[rollout_id][problem_id] = "X"
                    await asyncio.get_running_loop().run_in_executor(
                        None, self._write_sync
                    )

    def cleanup(self) -> None:
        try:
            if self.filepath.exists():
                self.filepath.unlink()
        except Exception:
            pass

async def generate_responses(
    args: argparse.Namespace,
    dataset_name: str,
    rollout_n: int,
    ports: List[int],
    logger: logging.Logger,
    semaphores: Dict[int, asyncio.Semaphore],
) -> None:
    """
    Asynchronously generate responses and save to output.jsonl.
    Implementation: Read existing output.jsonl to build cache, only generate missing entries.
    Generated results are appended to output.jsonl in real-time.
    """
    dataset_dir = Path(args.result_dir) / dataset_name
    output_file = dataset_dir / "output.jsonl"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    with StageContext(logger, f"C[{dataset_name}].1", "Reading cached output"):
        generated_results: List[Dict[str, Any]] = []
        cache: Set[Tuple[int, int]] = set()

        if output_file.exists():
            with output_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if (
                            "problem_id" in data
                            and "rollout_id" in data
                            and "response" in data
                            and data["response"] != ""
                        ):
                            generated_results.append(data)
                            cache.add((data["problem_id"], data["rollout_id"]))
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON line in output.jsonl, skipped.")

        logger.info("Loaded cache entries: %d", len(generated_results))

    with StageContext(logger, f"C[{dataset_name}].2", "Preparing generation tasks"):
        ds = load_dataset_from_hf(dataset_name)
        # max_concurrent_per_dp and semaphores are now handled externally and passed in

        tasks_to_process: List[Tuple[int, int, str, int]] = []
        ports_cycle = len(ports)

        for idx, sample in enumerate(ds):
            prompt = prepare_prompt(dataset_name, sample)
            for rollout_id in range(rollout_n):
                if (idx, rollout_id) in cache:
                    continue
                # port_idx = idx % ports_cycle
                port_idx = (idx * rollout_n + rollout_id) % ports_cycle
                tasks_to_process.append((idx, rollout_id, prompt, port_idx))

        logger.info("New requests to generate: %d", len(tasks_to_process))

        visualizer = ProgressVisualizer(
            dataset_dir / "process.txt", len(ds), rollout_n, cache
        )

        if not tasks_to_process:
            logger.info("All requests exist in cache, no generation needed.")
            visualizer.cleanup()
            return

    with StageContext(logger, f"C[{dataset_name}].3", "Parallel Generation"):
        file_lock = asyncio.Lock()

        async def generate_one_task(
            problem_id: int,
            rollout_id: int,
            prompt: str,
            port_idx: int,
            session: aiohttp.ClientSession,
        ) -> None:
            port = ports[port_idx]
            semaphore = semaphores[port]
            response = ""

            async with semaphore:
                try:
                    response = await generate_with_vllm_async(
                        session, prompt, port, args
                    )
                except Exception as exc:
                    logger.error(
                        "Generation failed problem=%06d rollout=%03d port=%d: %s",
                        problem_id,
                        rollout_id,
                        port,
                        exc,
                    )
                    return

            record = {
                "problem_id": problem_id,
                "rollout_id": rollout_id,
                "response": response,
            }

            generated_results.append(record)

            async with file_lock:
                with output_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            await visualizer.update(problem_id, rollout_id)

        connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                generate_one_task(pid, rid, pmt, pidx, session)
                for pid, rid, pmt, pidx in tasks_to_process
            ]
            await asyncio.gather(*tasks)
            visualizer.cleanup()

        logger.info("Dataset %s generation complete, results saved to %s", dataset_name, output_file)

def evaluate_dataset_results(
    args: argparse.Namespace,
    dataset_name: str,
    rollout_n: int,
    logger: logging.Logger,
) -> Dict[str, Dict[int, float]]:
    """
    Evaluation stage: Read output.jsonl, score and generate result.jsonl, return stats metrics.
    """
    dataset_dir = Path(args.result_dir) / dataset_name
    output_file = dataset_dir / "output.jsonl"
    result_file = dataset_dir / "result.jsonl"
    result_json_file = dataset_dir / "result.json"

    with StageContext(logger, f"D[{dataset_name}].1", "Loading model output"):
        if not output_file.exists():
            raise ValueError(f"output.jsonl not found, cannot evaluate: {dataset_name}")

        outputs_map: Dict[int, List[Tuple[int, str]]] = {}
        with output_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if "problem_id" in d and "rollout_id" in d:
                        outputs_map.setdefault(d["problem_id"], []).append(
                            (d["rollout_id"], d.get("response", ""))
                        )
                except json.JSONDecodeError:
                    pass

    with StageContext(logger, f"D[{dataset_name}].2", "Loading original dataset"):
        ds = load_dataset_from_hf(dataset_name)

    with StageContext(logger, f"D[{dataset_name}].3", "Parallel Evaluation & Metrics"):
        records_for_metrics: List[Dict[str, Any]] = []
        raw_stats_list: List[Dict[str, Any]] = []

        with result_file.open("w", encoding="utf-8") as rf:
            for idx, sample in enumerate(ds):
                problem_id = idx
                prompt = prepare_prompt(dataset_name, sample)

                rollouts = outputs_map.get(problem_id, [])
                # Sort by rollout_id
                rollouts.sort(key=lambda x: x[0])
                rollout_dict = {r[0]: r[1] for r in rollouts}

                responses = []
                scores = []

                for rid in range(rollout_n):
                    if rid not in rollout_dict:
                        raise ValueError(
                            f"Missing result: problem_id={problem_id} rollout_id={rid}. Please check if generation requests failed."
                        )
                    resp = rollout_dict.get(rid, "")
                    responses.append(resp)

                    if resp:
                        try:
                            s_res = score_response(dataset_name, resp, sample)
                            if isinstance(s_res, tuple):
                                score = float(s_res[0])
                            else:
                                score = float(s_res)
                        except Exception as e:
                            logger.warning("Scoring error p=%d r=%d: %s", problem_id, rid, e)
                            score = 0.0
                    else:
                        score = 0.0
                    scores.append(score)

                    records_for_metrics.append(
                        {"problem_id": problem_id, "rollout_id": rid, "score": score}
                    )

                if scores:
                    avg_val = statistics.mean(scores)
                    max_val = max(scores)
                    min_val = min(scores)
                    try:
                        std_val = statistics.stdev(scores)
                    except statistics.StatisticsError:
                        std_val = 0.0
                else:
                    avg_val = max_val = min_val = std_val = 0.0

                record = {
                    "problem_id": problem_id,
                    "prompt": prompt,
                    "responses": responses,
                    "scores": scores,
                    "avg": avg_val,
                    "max": max_val,
                    "min": min_val,
                    "std": std_val,
                    "data_source": dataset_name,
                }
                rf.write(json.dumps(record, ensure_ascii=False) + "\n")

                raw_stats_list.append(
                    {
                        "problem_id": problem_id,
                        "avg": avg_val,
                        "max": max_val,
                        "min": min_val,
                        "std": std_val,
                    }
                )

    with StageContext(logger, f"D[{dataset_name}].4", "Summarizing and writing files"):
        if raw_stats_list:
            summary = {
                "avg": statistics.mean(x["avg"] for x in raw_stats_list),
                "max": statistics.mean(x["max"] for x in raw_stats_list),
                "min": statistics.mean(x["min"] for x in raw_stats_list),
                "std": statistics.mean(x["std"] for x in raw_stats_list),
            }
        else:
            summary = {"avg": 0.0, "max": 0.0, "min": 0.0, "std": 0.0}

        final_json = {
            "data_source": dataset_name,
            "rollout_n": rollout_n,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": summary,
            "raw": raw_stats_list,
            "response_example": [
                outputs_map[0][0] if outputs_map else [],
            ],
        }

        with result_json_file.open("w", encoding="utf-8") as f:
            json.dump(final_json, f, indent=2, ensure_ascii=False)

        logger.info("Evaluation complete, results written to %s and %s", result_file, result_json_file)