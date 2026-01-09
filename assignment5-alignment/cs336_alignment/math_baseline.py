from vllm import LLM, SamplingParams
from typing import Callable, List, Any
from datasets import load_dataset, Dataset
from drgrpo_grader import r1_zero_reward_fn
from utils import evaluate_vllm
from pathlib import Path
import argparse
import json
import os
import torch

def load_math_dataset(path: os.PathLike) -> list[dict[str, str]]:
    examples = []
    with open(path, "r") as f:
        for line in f:
            example = json.loads(line)
            examples.append(example)
    return examples

def generate_prompts(dataset: list[Any]) -> List[str]:
    with open("prompts/r1_zero.prompt") as f:
        prompt_template = f.read()
    prompts = [
        prompt_template.format(question=example["problem"])
        for example in dataset
    ]
    return prompts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--input", default="MATH/validation.jsonl")
    parser.add_argument("--output", default="eval_outputs/math_baseline.jsonl")
    parser.add_argument("--n_samples", type=int, default=None)

    args = parser.parse_args()

    llm = LLM(
        args.model,
        max_model_len=2048,
        dtype=torch.bfloat16,
    )
    dataset = load_math_dataset(args.input)
    if args.n_samples is not None:
        dataset = dataset[:args.n_samples]
    prompts = generate_prompts(dataset)
    ground_truths = [example["solution"] for example in dataset]
    sampleing_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=2048,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    results = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampleing_params,
        output_path=Path(args.output),
    )
    print(results)