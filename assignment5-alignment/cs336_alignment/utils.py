from vllm import LLM, SamplingParams
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from typing import Callable
from pathlib import Path
import json
import torch

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str],
    eval_sampling_params: SamplingParams,
    output_path: Path,
) -> dict[str, float]:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """

    outputs = vllm_model.generate(prompts, eval_sampling_params)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {"format_reward": 0., "answer_reward": 0., "reward": 0.}
    with open(output_path, "w") as f:
        for prompt, output, ground_truth in zip(prompts, outputs, ground_truths):
            response = output.outputs[0].text

            eval_result = reward_fn(response, ground_truth)
            eval_result["prompt"] = prompt
            eval_result["ground_truth"] = ground_truth
            eval_result["response"] = response
            f.write(json.dumps(eval_result) + "\n")

            results["format_reward"] += eval_result["format_reward"] / len(prompts)
            results["answer_reward"] += eval_result["answer_reward"] / len(prompts)
            results["reward"] += eval_result["reward"] / len(prompts)
    return results

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
        encoded_inputs,
        padding=True,
        return_tensors="pt",
    )["input_ids"]
    input_ids = encoded_inputs[:, :-1]
    labels = encoded_inputs[:, 1:]
    response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
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

def compute_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdims=True)
    log_probs_for_labels = torch.gather(log_probs, -1, labels.unsqueeze(-1))
    return log_probs_for_labels.squeeze(-1)

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits
    result = {"log_probs": compute_log_probs(logits, labels)}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    return result

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    tensor = torch.masked_fill(tensor, ~mask, 0)
    if dim is not None:
        sum = torch.sum(tensor, dim=dim)
    else:
        sum = torch.sum(tensor)
    return sum / normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss = -masked_normalize(
        policy_log_probs,
        response_mask,
        normalize_constant,
    ) / (policy_log_probs.shape[0] * gradient_accumulation_steps)
    
    loss.backward()
    return loss.item(), {}

def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    rewards = []
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        rewards.append(reward_fn(response, ground_truth)["reward"])
    
    rewards = torch.Tensor(rewards).view(-1, group_size)
    mean_rewards = rewards.mean(dim=-1, keepdims=True)
    advantages = rewards - mean_rewards
    if normalize_by_std:
        std_rewards = rewards.std(dim=-1, keepdims=True)
        advantages = advantages / (std_rewards + advantage_eps)
    return advantages.view(-1), rewards.view(-1), {}
