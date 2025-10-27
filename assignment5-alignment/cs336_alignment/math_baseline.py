from typing import Callable
from collections import defaultdict

from vllm import LLM, SamplingParams
import pandas as pd
import pyarrow.parquet as pq

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str],
    eval_sampling_params: SamplingParams,
) -> None:
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    total_rewards = defaultdict(float)
    for output, ground_truth in zip(outputs, ground_truths):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        rewards = reward_fn(generated_text, ground_truth)
        for k, v in rewards.items():
            total_rewards[k] += v
    print(total_rewards)

if __name__ == "__main__":
    with open("cs336_alignment/prompts/r1_zero.prompt", "r") as f:
        template = f.read()
    df = pd.read_parquet("data/gsm8k_main_test.parquet", engine="pyarrow")
    prompts = [template.replace("{question}", question) for question in df["question"]]
    ground_truths = df["answer"]

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    llm = LLM(model="Qwen/Qwen2.5-Math-1.5B")
    evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
    )