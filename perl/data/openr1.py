import re
import math

from typing import Optional
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from datasets import load_dataset

from .system_prompts import make_conversation


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""

    def count_tags(text: str) -> float:
        count = 0.0
        # We only count </think> tag, because <think> tag is available in system prompt
        if text.count("\n</think>\n") == 1:
            count += 1.0
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


def get_cosine_scaled_reward(
        min_value_wrong: float = -0.0,
        max_value_wrong: float = -0.5,
        min_value_correct: float = 0.5,
        max_value_correct: float = 1.0,
        max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            try:
                # Try to parse gold solution with timeout protection
                gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
                if len(gold_parsed) == 0:
                    # Skip unparseable examples with neutral reward
                    rewards.append(0.0)
                    continue

                # Try to parse model answer with timeout protection
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                boxed=True,
                                units=True,
                            ),
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )

                # Verify answer with timeout protection
                is_correct = verify(answer_parsed, gold_parsed)
                gen_len = len(content)

                # Apply cosine scaling based on length
                progress = min(gen_len / max_len, 1.0)  # Clamp to [0, 1]
                cosine = math.cos(progress * math.pi)

                if is_correct:
                    min_value = min_value_correct
                    max_value = max_value_correct
                else:
                    # Swap min/max for incorrect answers
                    min_value = max_value_wrong
                    max_value = min_value_wrong

                reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
                rewards.append(float(reward))
                
            except Exception as e:
                # Catch all exceptions including TimeoutException
                # Return neutral reward to avoid blocking training
                rewards.append(0.0)

        return rewards

    return cosine_scaled_reward


def load_openr1_dataset(dataset_name_or_path: str, example_numbers: int = None):
    dataset = load_dataset(
        dataset_name_or_path, split="train"
    )

    dataset = dataset.map(make_conversation)

    if example_numbers is not None and len(dataset) > example_numbers:
        dataset = dataset.select(range(example_numbers))

    return {
        "train_dataset": dataset,
        "test_dataset": None,
        "reward_functions": [format_reward, get_cosine_scaled_reward()],
        "reward_weights": [1.0, 2.0]
    }