from typing import Callable
from collections import defaultdict
import json

from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.datasets import GSM8KTestDataset

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str],
    eval_sampling_params: SamplingParams,
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """

    total_rewards = defaultdict(float)
    samples = []
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    for output, ground_truth in zip(outputs, ground_truths):
        generated_text = output.outputs[0].text
        
        sample = {
            "prompt": output.prompt,
            "generated_text": generated_text,
            "ground_truth": ground_truth,
        }
        rewards = reward_fn(generated_text, ground_truth)
        for k, v in rewards.items():
            sample[k] = v
            total_rewards[k] += v
        samples.append(sample)
    with open("data/math_baseline.json", "w") as f:
        json.dump(samples, f, indent=4)
    print(f"=== Total rewards for {len(prompts)} samples ===")
    print(f" - format_reward: {total_rewards['format_reward']}")
    print(f" - answer_reward: {total_rewards['answer_reward']}")
    print(f" - reward: {total_rewards['reward']}")

if __name__ == "__main__":
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    llm = LLM(model="Qwen/Qwen2.5-Math-1.5B")
    ds = GSM8KTestDataset()
    evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=ds.prompts,
        ground_truths=ds.ground_truths,
        eval_sampling_params=sampling_params,
    )