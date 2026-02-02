from drgrpo_grader import r1_zero_reward_fn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from torch.utils.data import Dataset, DataLoader, Subset
from unittest.mock import patch
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

import os
import json
import random
import torch
import wandb

from utils import (
    aggregate_entropy,
    get_response_log_probs,
    tokenize_prompt_and_output,
    evaluate_vllm,
    sft_microbatch_train_step,
    compute_group_normalized_rewards,
    grpo_microbatch_train_step,
)

class SFTDataset(Dataset):
    def __init__(self, file_path: os.PathLike):
        with open(file_path, "r") as f:
            self.data = [json.loads(line) for line in f]
        self.data = [sample for sample in self.data]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]["prompt"], self.data[index]["response"]

class EIDataset(Dataset):
    def __init__(self, rollouts: list[dict[str, str]]):
        self.data = []
        for rollout in rollouts:
            rewards = r1_zero_reward_fn(rollout["response"], rollout["ground_truth"])
            if rewards["reward"] > 0.: 
                self.data.append(rollout)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]["prompt"], self.data[index]["response"]

class GRPODataset(Dataset):
    def __init__(
        self,
        rollouts: list[dict[str, str]],
    ):
        self.data = rollouts

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return (
            self.data[index]["prompt"],
            self.data[index]["response"],
            self.data[index]["ground_truth"],
        )

def init_vllm(
    model_id: str,
    device: str,
    seed: int,
    gpu_memory_utilization: float = 0.85,
):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    13
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def init(config):
    run = wandb.init(
        project=config["wandb_project"],
        name=config["exp_id"],
        config=config,
    )

    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    return run

def load_from_raw_dataset(path, n_samples):
    with open("prompts/r1_zero.prompt") as f:
        prompt_template = f.read()
    samples = []
    with open(path, "r") as f:
        for line in f:
            samples.append(json.loads(line))
    random.shuffle(samples)
    samples = samples[:n_samples]
    
    prompts = [prompt_template.format(question=sample["problem"]) for sample in samples]
    ground_truths = [sample["solution"] for sample in samples]
    return {
        "prompts": prompts,
        "ground_truths": ground_truths,
    }


def evaluate(config, policy, llm, eval_data, eval_step, train_step):
    load_policy_into_vllm_instance(policy, llm)
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=config["max_seq_len"],
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    results = evaluate_vllm(
        llm,
        r1_zero_reward_fn,
        eval_data["prompts"],
        eval_data["ground_truths"],
        sampling_params,
        Path("eval_outputs") / config["exp_id"] / f"step_{eval_step}.jsonl",
    )
    wandb.log({
        "eval/format_reward": results["format_reward"],
        "eval/answer_reward": results["answer_reward"],
        "eval/reward": results["reward"],
        "eval/train_step_at_eval": train_step,
        "eval_step": eval_step,
    })


def train(model, optimizer, tokenizer, dataloader, config, train_step):
    optimizer.zero_grad()
    train_loss = 0.
    token_entropy = 0.
    micro_step = 0
    for _ in tqdm(range(config["n_epochs"]), "epoch"):
        for micro_batch in tqdm(dataloader, "micro_batch_id"):
            micro_step += 1

            prompts, responses = micro_batch
            micro_batch = tokenize_prompt_and_output(prompts, responses, tokenizer)
            micro_batch = {
                k:v[:,:config["max_seq_len"]].to(config["train_device"])
                for k,v in micro_batch.items()
            }

            results = get_response_log_probs(
                model,
                micro_batch["input_ids"],
                micro_batch["labels"],
                return_token_entropy=True,
            )
            token_entropy += aggregate_entropy(
                results["token_entropy"],
                micro_batch["response_mask"],
                config["gradient_accumulation_steps"],
            )

            loss, _ = sft_microbatch_train_step(
                results["log_probs"],
                micro_batch["response_mask"],
                config["gradient_accumulation_steps"],
            )
            train_loss += loss

            if micro_step % config["gradient_accumulation_steps"] == 0:
                train_step += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                wandb.log({
                    "train/loss": train_loss,
                    "train/token_entropy": token_entropy,
                    "train_step": train_step,
                })
                train_loss = 0.
                token_entropy = 0.

    return train_step

def main_sft():
    config = {
        "wandb_project": "cs336-assignment5-sft",
        "exp_id": "sft-samples-exp",
        "seed": 42,
        "model_id": "Qwen/Qwen2.5-Math-1.5B",
        "train_device": "cuda:1",
        "inference_device": "cuda:0",
        "train_data": "MATH/sft.jsonl",
        "eval_data": "MATH/validation.jsonl",
        "lr": 1e-5,
        "weight_decay": 1e-5,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "eval_steps": 64,
        "n_epochs": 1,
        "n_sft_samples": 256,
        "n_eval_samples": 100,
        "max_seq_len": 40960,
    }

    llm = init_vllm(
        config["model_id"],
        config["inference_device"],
        config["seed"],
        0.65,
    )
    eval_data = load_from_raw_dataset(config["eval_data"], config["n_eval_samples"])
    for lr in [1e-4, 1e-5, 1e-6]:
        for gradient_accumulation_steps in [16, 32, 64]:
            for n_sft_samples in [128, 256, 512, 1024]:
                config["n_sft_samples"] = n_sft_samples
                config["lr"] = lr
                config["gradient_accumulation_steps"] = gradient_accumulation_steps
                config["exp_id"] = f"lr={lr}, batch_size={gradient_accumulation_steps}, n_sft_samples={n_sft_samples}"
                run = init(config)

                model = AutoModelForCausalLM.from_pretrained(
                    config["model_id"],
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                ).to(config["train_device"])
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=config["lr"],
                    weight_decay=config["weight_decay"],
                )
                tokenizer = AutoTokenizer.from_pretrained(config["model_id"])

                dataset = SFTDataset(config["train_data"])
                if config["n_sft_samples"]:
                    dataset = Subset(dataset, range(config["n_sft_samples"]))
                dataloader = DataLoader(
                    dataset,
                    batch_size=config["micro_batch_size"],
                    shuffle=True,
                )

                evaluate(config, model, llm, eval_data, 0, 0)
                train_step = train(model, optimizer, tokenizer, dataloader, config, 0)
                evaluate(config, model, llm, eval_data, 1, train_step)
                run.finish()

def sample_rollouts(
    policy,
    llm,
    config,
    prompt_data_path,
    n_prompts,
    rollout_data_path,
    n_rollouts_per_prompt,
):
    data = load_from_raw_dataset(
        prompt_data_path,
        n_prompts,
    )

    load_policy_into_vllm_instance(policy, llm)
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=config["sampling_max_tokens"],
        min_tokens=config["sampling_min_tokens"],
        stop=["</answer>"],
        include_stop_str_in_output=True,
        n=n_rollouts_per_prompt,
    )

    outputs = llm.generate(data["prompts"], sampling_params)
    rollouts = []
    with open(rollout_data_path, "w") as f:
        for prompt, ground_truth, completion in zip(data["prompts"], data["ground_truths"], outputs):
            for response in completion.outputs:
                rollout = {
                    "prompt": prompt,
                    "ground_truth": ground_truth,
                    "response": response,
                }
                rollouts.append(rollout)
                f.write(json.dumps(rollout) + "\n")
    return rollouts

def main_ei():
    config = {
        "wandb_project": "cs336-assignment5-ei",
        "exp_id": "ei",
        "seed": 42,
        "model_id": "Qwen/Qwen2.5-Math-1.5B",
        "train_device": "cuda:1",
        "inference_device": "cuda:0",
        "train_data": "MATH/train.jsonl",
        "eval_data": "MATH/validation.jsonl",
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 32,
        "eval_steps": 64,
        "n_epochs": 1,
        "n_sft_samples": 0,
        "n_eval_samples": 100,
        "max_seq_len": 40960,
        "n_ei_steps": 5,
        "sampling_min_tokens": 4,
        "sampling_max_tokens": 40960,
    }

    llm = init_vllm(
        config["model_id"],
        config["inference_device"],
        config["seed"],
        0.65,
    )

    hparams_to_sweep = [
        [1024, 1, 3],
        [512, 2, 3],
        [512, 1, 6],
        [256, 4, 3],
        [256, 1, 12],
        [256, 2, 6],
    ]
    for n_ei_samples, n_ei_rollouts, n_epochs in hparams_to_sweep:
        config["n_ei_samples"] = n_ei_samples
        config["n_ei_rollouts"] = n_ei_rollouts
        config["n_epochs"] = n_epochs
        config["exp_id"] = f"n_ei_samples={n_ei_samples},n_ei_rollouts={n_ei_rollouts},n_epochs={n_epochs}"
        run = init(config)

        model = AutoModelForCausalLM.from_pretrained(
            config["model_id"],
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(config["train_device"])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )
        tokenizer = AutoTokenizer.from_pretrained(config["model_id"])

        eval_data = load_from_raw_dataset(config["eval_data"], config["n_eval_samples"])
        evaluate(config, model, llm, eval_data, 0, 0)

        train_step = 0
        for ei_step in tqdm(range(config["n_ei_steps"]), "ei_step"):
            rollouts = sample_rollouts(
                model,
                llm,
                config,
                config["train_data"],
                config["n_ei_samples"],
                Path("ei_rollouts") / config["exp_id"] / f"ei_step={ei_step}",
                config["n_ei_rollouts"],
            )
            dataset = EIDataset(rollouts)
            dataloader = DataLoader(
                dataset,
                batch_size=config["micro_batch_size"],
                shuffle=True,
            )
            train_step = train(model, optimizer, tokenizer, dataloader, config, train_step)
            evaluate(config, model, llm, eval_data, ei_step+1, train_step)
        run.finish()


def train_grpo(model, optimizer, tokenizer, dataloader, config, train_step):
    model.eval()
    batches = []
    for batch in tqdm(dataloader, "batch_id"):
        prompts, responses, ground_truths = batch
        batch = tokenize_prompt_and_output(prompts, responses, tokenizer)
        batch = {
            k:v[:,:config["max_seq_len"]].to(config["train_device"])
            for k,v in batch.items()
        }

        advantages, rewards, metadata = compute_group_normalized_rewards(
            r1_zero_reward_fn,
            responses,
            ground_truths,
            config["group_size"],
            config["advantage_eps"],
            config["use_std_normalization"],
        )
        batch["advantages"] = advantages.view(-1, 1).to(config["train_device"])
        batch["rewards"] = rewards.view(-1, 1).to(config["train_device"])
        for k,v in metadata:
            batch[k] = v.to(config["train_device"])
        with torch.no_grad():
            results = get_response_log_probs(
                model,
                batch["input_ids"],
                batch["labels"],
                return_token_entropy=False,
            )
            batch["old_log_probs"] = results["log_probs"].to(config["train_device"])
        batches.append(batch)

    model.train()
    optimizer.zero_grad()
    train_loss = 0.
    token_entropy = 0.
    micro_step = 0

    for _ in tqdm(range(config["n_epochs"]), "epoch"):
        for batch in batches:
            micro_step += 1
            results = get_response_log_probs(
                model,
                micro_batch["input_ids"],
                micro_batch["labels"],
                return_token_entropy=True,
            )
            token_entropy += aggregate_entropy(
                results["token_entropy"],
                micro_batch["response_mask"],
                config["gradient_accumulation_steps"],
            )

            loss, _ = grpo_microbatch_train_step(
                results["log_probs"],
                micro_batch["response_mask"],
                config["gradient_accumulation_steps"],
                config["loss_type"],
                rewards,
                advantages,
                micro_batch["old_log_probs"],
                config["grpo_clip_range"],
            )
            train_loss += loss

            if micro_step % config["gradient_accumulation_steps"] == 0:
                train_step += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                wandb.log({
                    "train/loss": train_loss,
                    "train/token_entropy": token_entropy,
                    "train_step": train_step,
                })
                train_loss = 0.
                token_entropy = 0.

    return train_step

def main_grpo():
    config = {
        "wandb_project": "cs336-assignment5-grpo",
        "exp_id": "grpo",
        "seed": 42,
        "model_id": "Qwen/Qwen2.5-Math-1.5B",
        "train_device": "cuda:1",
        "inference_device": "cuda:0",
        "train_data": "MATH/train.jsonl",
        "eval_data": "MATH/validation.jsonl",
        "n_grpo_steps": 200,
        "advantage_eps": 1e-6,
        "grpo_clip_range": 0.2,
        "rollout_batch_size": 256,
        "group_size": 8,
        "sampling_min_tokens": 15,
        "sampling_max_tokens": 40960,
        "epochs_per_rollout_batch": 1,
        "train_batch_size": 256,
        "lr": 1e-5,
        "weight_decay": 0.0,
        "gradient_accumulation_steps": 128,
        "gpu_memory_utilization": 0.85,
        "loss_type": "reinforce_with_baseline",
        "use_std_normalization": True,
        "eval_grpo_steps": 2,
        "n_eval_samples": 1024,
        "max_seq_len": 40960,
    }

    llm = init_vllm(
        config["model_id"],
        config["inference_device"],
        config["seed"],
        config["gpu_memory_utilization"],
    )

    run = init(config)
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(config["train_device"])
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    eval_data = load_from_raw_dataset(config["eval_data"], config["n_eval_samples"])
    evaluate(config, model, llm, eval_data, 0, 0)

    train_step = 0
    for grpo_step in tqdm(range(config["n_grpo_steps"]), "grpo_step"):
        rollouts = sample_rollouts(
            model,
            llm,
            config,
            config["grpo_train_data"],
            config["rollout_batch_size"] // config["group_size"],
            Path("grpo_rollouts") / config["exp_id"] / f"grpo_step={grpo_step}.jsonl",
            config["group_size"],
        )
        dataset = GRPODataset(rollouts)
        dataloader = DataLoader(
            dataset,
            batch_size=config["train_batch_size"],
            shuffle=False,
        )
        train_step = train_grpo(model, optimizer, tokenizer, llm, dataloader, config, train_step)
        if (grpo_step+1) % config["eval_grpo_steps"] == 0:
            evaluate(config, model, llm, eval_data, grpo_step+1, train_step)
    run.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--method", type=str, default="sft")

    args = parser.parse_args()
    if args.method == "sft":
        main_sft()
    elif args.method == "ei":
        main_ei()
    elif args.method == "grpo":
        main_grpo()
    else:
        print(f"Unsupported method {args.method}!")
        