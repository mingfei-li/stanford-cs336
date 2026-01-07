from vllm import LLM, SamplingParams
from typing import Callable, List
from datasets import load_dataset, Dataset
from drgrpo_grader import r1_zero_reward_fn
from utils import evaluate_vllm
import json

def load_math_dataset() -> list[dict[str, str]]:
    examples = []
    with open("MATH/validation.jsonl") as f:
        for line in f:
            example = json.loads(line)
            examples.append(example)
    return examples

def generate_prompts(dataset: Dataset) -> List[str]:
    with open("prompts/r1_zero.prompt") as f:
        prompt_template = f.read()
    prompts = [
        prompt_template.format(question=example["problem"])
        for example in dataset
    ]
    return prompts

if __name__ == "__main__":
    llm = LLM("Qwen/Qwen2.5-Math-1.5B")
    dataset = load_math_dataset()
    prompts = generate_prompts(dataset)
    ground_truths = [example["solution"] for example in dataset]
    sampleing_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    evaluate_vllm(
        eval_id="math_baseline",
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampleing_params
    )