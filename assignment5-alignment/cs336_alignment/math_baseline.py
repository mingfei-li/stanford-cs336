from vllm import LLM, SamplingParams
from typing import Callable, List
from datasets import load_dataset, Dataset
from drgrpo_grader import r1_zero_reward_fn
import json

def load_math_dataset() -> list[dict[str, str]]:
    examples = []
    with open("MATH/validation.jsonl") as f:
        for line in f:
            example = json.loads(line)
            examples.append(example)
    return examples

def evaluate_vllm(
    eval_id: str,
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],
    eval_sampling_params: SamplingParams,
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """

    outputs = vllm_model.generate(prompts, eval_sampling_params)
    with open(f"eval_outputs/{eval_id}.jsonl", "w") as f:
        for prompt, output, ground_truth in zip(prompts, outputs, ground_truths):
            response = output.outputs[0].text
            eval_result = reward_fn(response, ground_truth)
            eval_result["prompt"] = prompt
            eval_result["ground_truth"] = ground_truth
            eval_result["response"] = response
            f.write(json.dumps(eval_result) + "\n")

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