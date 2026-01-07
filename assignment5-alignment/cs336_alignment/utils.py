from vllm import LLM, SamplingParams
from transformers import PreTrainedTokenizerBase
from typing import Callable
import json
import torch

def evaluate_vllm(
    eval_id: str,
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

    outputs = vllm_model.generate(prompts, eval_sampling_params)
    with open(f"eval_outputs/{eval_id}.jsonl", "w") as f:
        for prompt, output, ground_truth in zip(prompts, outputs, ground_truths):
            response = output.outputs[0].text
            eval_result = reward_fn(response, ground_truth)
            eval_result["prompt"] = prompt
            eval_result["ground_truth"] = ground_truth
            eval_result["response"] = response
            f.write(json.dumps(eval_result) + "\n")

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    prompt_ids = tokenizer(prompt_strs)["input_ids"]
    output_ids = tokenizer(output_strs)["input_ids"]
    encoded_inputs = {
        "input_ids": [
            prompt + output 
            for prompt, output in zip(prompt_ids, output_ids)
        ],
    }
    encoded_inputs = tokenizer.pad(
        encoded_inputs, padding=True, return_tensors="pt")["input_ids"]
    input_ids = encoded_inputs[:, :-1]
    labels = encoded_inputs[:, 1:]
    response_mask = torch.zeros_like(input_ids)
    for i in range(len(prompt_ids)):
        start = len(prompt_ids[i]) - 1
        end = start + len(output_ids[i])
        response_mask[i, start:end] = 1
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    max_logits, _ = torch.max(logits, dim=-1, keepdims=True)
    logits = logits - max_logits
    exps = torch.exp(logits)
    sum_exps = torch.sum(exps, dim=-1, keepdims=True)
    probs = exps / sum_exps
    log_probs = logits - torch.log(sum_exps)
    return -torch.sum(probs * log_probs, dim=-1)