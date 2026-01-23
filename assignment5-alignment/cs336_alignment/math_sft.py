from drgrpo_grader import r1_zero_reward_fn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from torch.utils.data import Dataset, DataLoader, Subset
from unittest.mock import patch
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from tqdm import tqdm
from pathlib import Path

import os
import json
import random
import torch
import wandb

from utils import (
    get_response_log_probs,
    masked_normalize,
    tokenize_prompt_and_output,
    evaluate_vllm,
    sft_microbatch_train_step,
)

class SFTDataset(Dataset):
    def __init__(self, file_path: os.PathLike):
        with open(file_path, "r") as f:
            self.data = [json.loads(line) for line in f]
        self.data = [sample for sample in self.data if sample.get("reward", 1.0) == 1.0]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]["prompt"], self.data[index]["response"]

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


def evaluate(config, policy, llm, eval_data, step):
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
        Path("eval_outputs") / config["exp_id"] / f"step_{step}.jsonl",
    )
    wandb.log({
        "eval/format_reward": results["format_reward"],
        "eval/answer_reward": results["answer_reward"],
        "eval/reward": results["reward"],
    }, step=step)


def train(model, tokenizer, llm, dataloader, config, global_step, ei_step=None):
    eval_data = load_from_raw_dataset(config["eval_data"], config["n_eval_samples"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    optimizer.zero_grad()
    evaluate(config, model, llm, eval_data, global_step)
    train_loss = 0.
    local_step = 0
    for epoch in tqdm(range(config["n_epochs"]), "epoch"):
        for batch_id, batch in enumerate(tqdm(dataloader, "batch_id")):
            global_step += 1
            local_step += 1

            prompts, responses = batch
            batch = tokenize_prompt_and_output(prompts, responses, tokenizer)
            batch = {k:v[:,:config["max_seq_len"]].to(config["train_device"]) for k,v in batch.items()}

            results = get_response_log_probs(
                model,
                batch["input_ids"],
                batch["labels"],
                return_token_entropy=True,
            )

            loss, _ = sft_microbatch_train_step(
                results["log_probs"],
                batch["response_mask"],
                config["gradient_accumulation_steps"],
            )
            train_loss += loss

            if local_step % config["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                wandb.log({
                    "train/loss": train_loss,
                    "train/token_entropy": results["token_entropy"].mean(),
                }, step=global_step)
                train_loss = 0.

            if local_step % config["eval_steps"] == 0:
                evaluate(config, model, llm, eval_data, global_step)

    return global_step

def train_sft():
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
                tokenizer = AutoTokenizer.from_pretrained(config["model_id"])

                dataset = SFTDataset(config["train_data"])
                if config["n_sft_samples"]:
                    dataset = Subset(dataset, range(config["n_sft_samples"]))
                dataloader = DataLoader(
                    dataset,
                    batch_size=config["micro_batch_size"],
                    shuffle=True,
                )
                train(model, tokenizer, llm, dataloader, config, 0)
                run.finish()

def sample_rollouts(policy, llm, config):
    data = load_from_raw_dataset(
        config["ei_data"],
        config["n_ei_samples"],
    )

    load_policy_into_vllm_instance(policy, llm)
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=config["max_seq_len"],
        min_tokens=15,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        n=config["n_ei_rollouts"],
    )
    results = evaluate_vllm(
        llm,
        r1_zero_reward_fn,
        data["prompts"],
        data["ground_truths"],
        sampling_params,
        config["train_data"],
    )

def train_ei():
    config = {
        "wandb_project": "cs336-assignment5-ei",
        "exp_id": "ei",
        "seed": 42,
        "model_id": "Qwen/Qwen2.5-Math-1.5B",
        "train_device": "cuda:1",
        "inference_device": "cuda:0",
        "train_data": "MATH/sft.jsonl",
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
        "ei_data": "MATH/train.jsonl",
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
        tokenizer = AutoTokenizer.from_pretrained(config["model_id"])

        global_step = 0
        for ei_step in tqdm(range(config["n_ei_steps"]), "ei_step"):
            config["train_data"] = Path("ei_train_data") / config["exp_id"] / f"{ei_step}.jsonl"
            sample_rollouts(model, llm, config)
            dataset = SFTDataset(config["train_data"])
            dataloader = DataLoader(
                dataset,
                batch_size=config["micro_batch_size"],
                shuffle=True,
            )
            global_step = train(model, tokenizer, llm, dataloader, config, global_step, ei_step)
        run.finish()

if __name__ == "__main__":
    #train_sft()
    train_ei()