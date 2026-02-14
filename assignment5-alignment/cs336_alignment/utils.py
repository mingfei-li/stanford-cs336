from math import e
from vllm import LLM, SamplingParams
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from typing import Callable, Literal
from pathlib import Path
from collections import defaultdict
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

    n_rollouts = len(prompts) * eval_sampling_params.n
    results = {"format_reward": 0., "answer_reward": 0., "reward": 0., "response_length": 0.}
    with open(output_path, "w") as f:
        for prompt, outputs_for_prompt, ground_truth in zip(prompts, outputs, ground_truths):
            for output in outputs_for_prompt.outputs:
                response = output.text
                eval_result = reward_fn(response, ground_truth)
                eval_result["prompt"] = prompt
                eval_result["ground_truth"] = ground_truth
                eval_result["response"] = response
                f.write(json.dumps(eval_result) + "\n")

                results["format_reward"] += eval_result["format_reward"] / n_rollouts
                results["answer_reward"] += eval_result["answer_reward"] / n_rollouts
                results["reward"] += eval_result["reward"] / n_rollouts
                results["response_length"] += len(response) / n_rollouts
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
        with torch.no_grad():
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

def get_per_response_entropy(
    token_entropy: torch.Tensor,
    response_mask: torch.Tensor,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    return masked_normalize(
        token_entropy,
        response_mask,
        normalize_constant,
        dim=-1,
    ) / response_mask.float().sum(dim=-1)

def aggregate_entropy(
    token_entropy: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    entropy = get_per_response_entropy(
        token_entropy,
        response_mask,
        normalize_constant,
    )
    entropy = entropy.mean() / gradient_accumulation_steps
    return entropy.item()


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    rewards = []
    format_rewards = []
    answer_rewards = []
    response_lens = []
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        results = reward_fn(response, ground_truth)
        rewards.append(results["reward"])
        format_rewards.append(results["format_reward"])
        answer_rewards.append(results["answer_reward"])
        response_lens.append(len(response))

    rewards = torch.Tensor(rewards).view(-1, group_size)
    mean_rewards = rewards.mean(dim=-1, keepdims=True)
    std_rewards = rewards.std(dim=-1, keepdims=True)
    advantages = rewards - mean_rewards
    if normalize_by_std:
        advantages = advantages / (std_rewards + advantage_eps)

    advantages = advantages.view(-1)
    rewards = rewards.view(-1)

    with torch.no_grad():
        metadata = {
            "avg_group_reward_std": std_rewards.mean(),
            "max_group_reward_std": std_rewards.max(),
            "min_group_reward_std": std_rewards.min(),
            "avg_rewards": rewards.mean(),
            "avg_format_rewards": torch.Tensor(format_rewards).mean(),
            "avg_answer_rewards": torch.Tensor(answer_rewards).mean(),
            "avg_response_length": torch.Tensor(response_lens).mean(),
            "rollout_batch_size": rewards.shape[0],
        }
        advantages_by_response_len_bucket = defaultdict(list)
        for response_len, advantage in zip(response_lens, advantages):
            bucket = response_len // 1000
            advantages_by_response_len_bucket[bucket].append(advantage.item())
        for bucket, advantages_for_bucket in advantages_by_response_len_bucket.items():
            metadata[f"avg_advantage_for_bucket_{bucket}"] = torch.Tensor(advantages_for_bucket).mean().item()

    return advantages, rewards, metadata

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    loss = -raw_rewards_or_advantages * policy_log_probs
    return loss

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    r = torch.exp(policy_log_probs - old_log_probs)
    loss = -torch.min(r * advantages, torch.clip(r, 1-cliprange, 1+cliprange) * advantages)
    with torch.no_grad():
        is_clipped = (r*advantages) > (torch.clip(r, 1-cliprange, 1+cliprange)*advantages)
        kl_term = old_log_probs - policy_log_probs
    return loss, {"is_clipped": is_clipped, "kl_term": kl_term}

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        assert raw_rewards is not None
        return compute_naive_policy_gradient_loss(
            raw_rewards,
            policy_log_probs,
        ), {}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None
        return compute_naive_policy_gradient_loss(
            advantages,
            policy_log_probs,
        ), {}
    else:
        assert loss_type == "grpo_clip"
        assert old_log_probs is not None
        assert cliprange is not None
        return compute_grpo_clip_loss(
            advantages,
            policy_log_probs,
            old_log_probs,
            cliprange,
        )

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    if dim is not None:
        return torch.sum(tensor*mask, dim=dim) / torch.sum(mask, dim=dim)
    else:
        return torch.sum(tensor*mask) / torch.sum(mask)

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baselien", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    normalize_constant: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange,
    )
    if normalize_constant is None:
        loss = masked_mean(loss, response_mask, dim=-1)
    else:
        loss = masked_normalize(loss, response_mask, normalize_constant, dim=-1) 
    if loss_type == "grpo_clip":
        clip_fraction = masked_mean(
            metadata["is_clipped"],
            response_mask,
            dim=-1,
        )
        kl_term = masked_normalize(
            metadata["kl_term"],
            response_mask,
            normalize_constant=1.0,
            dim=-1,
        )
        metadata = {
            "clip_fraction": clip_fraction,
            "kl_term": kl_term,
        }
    metadata["loss"] = loss.detach()
    loss = loss.mean() / gradient_accumulation_steps
    loss.backward()
    return loss, metadata
