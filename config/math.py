"""
name:
Math 500

dataset:
HuggingFaceH4/MATH-500

abstract:
This dataset contains a subset of 500 problems from the MATH benchmark that
OpenAI created in their Let's Verify Step by Step paper.

languages:
english

tags:
math, reasoning

paper:
https://arxiv.org/abs/2305.20050

starred:
true
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

import random


MATH_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly. The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}
""".strip()

# Prompt template from simple-evals: https://github.com/openai/simple-evals/blob/83ed7640a7d9cd26849bcb3340125002ef14abbe/common.py#L14
GPQA_QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

def aime_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["problem"]),
        choices=[line["answer"]],
        gold_index=0,
    )


def amc_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["problem"]),
        choices=[line["answer"]],
        gold_index=0,
    )


def math_500_prompt(line, task_name: str = None):
    query = MATH_QUERY_TEMPLATE.format(Question=line["problem"])
    return Doc(
        task_name=task_name,
        query=query,
        choices=[line["solution"]],
        gold_index=0,
    )

def minerva_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["problem"]),
        choices=[line["solution"]],
        gold_index=0,
    )

def olympiadbench_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["question"]),
        choices=[line["answer"]],
        gold_index=0,
    )


def gpqa_prompt_fn(line, task_name: str = None):
    gold_index = random.randint(0, 3)
    choices = [line["Incorrect Answer 1"], line["Incorrect Answer 2"], line["Incorrect Answer 3"]]
    choices.insert(gold_index, line["Correct Answer"])
    query = GPQA_QUERY_TEMPLATE.format(
        A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=line["Question"]
    )
    return Doc(
        task_name=task_name,
        query=query,
        choices=["A", "B", "C", "D"],
        gold_index=gold_index,
        instruction=query,
    )


math_500_avg_4 = LightevalTaskConfig(
    name="math_500_avg_4",
    prompt_function=math_500_prompt,
    hf_repo="HuggingFaceH4/MATH-500",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=16384,
    metrics=[Metrics.avg_at_n_math],
    stop_sequence=[],
)

minerva_pass_4 = LightevalTaskConfig(
    name="minerva_pass_4",
    prompt_function=minerva_prompt_fn,
    hf_repo="knoveleng/Minerva-Math",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=16384,
    metrics=[Metrics.pass_at_k_math],
    stop_sequence=[],
)

amc23_pass_32 = LightevalTaskConfig(
    name="amc23_pass_32",
    prompt_function=amc_prompt_fn,
    hf_repo="knoveleng/AMC-23",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=16384,
    metrics=[Metrics.pass_at_k_math],
    stop_sequence=[],
)

aime24_pass_32 = LightevalTaskConfig(
    name="aime24_pass_32",
    prompt_function=aime_prompt_fn,
    hf_repo="HuggingFaceH4/aime_2024",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=16384,
    metrics=[Metrics.pass_at_k_math],
    stop_sequence=[],
)

aime25_pass_32 = LightevalTaskConfig(
    name="aime25_pass_32",
    prompt_function=aime_prompt_fn,
    hf_repo="yentinglin/aime_2025",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=16384,
    metrics=[Metrics.pass_at_k_math],
    stop_sequence=[],
)

gpqa_diamond_pass_4 = LightevalTaskConfig(
    name="gpqa_diamond_pass_4",
    prompt_function=gpqa_prompt_fn,
    hf_repo="Idavidrein/gpqa",
    hf_subset="gpqa_diamond",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=16384,
    metrics=[Metrics.pass_at_k_math],
    stop_sequence=[],
)

olympiadbench_pass_4 = LightevalTaskConfig(
    name="olympiadbench_pass_4",
    prompt_function=olympiadbench_prompt_fn,
    hf_repo="knoveleng/OlympiadBench",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=16384,
    metrics=[Metrics.pass_at_k_math],
    stop_sequence=[],
)

# Add tasks to the table
TASKS_TABLE = [
    math_500_avg_4,
    minerva_pass_4,
    amc23_pass_32,
    aime24_pass_32,
    aime25_pass_32,
    gpqa_diamond_pass_4,
    olympiadbench_pass_4,
]