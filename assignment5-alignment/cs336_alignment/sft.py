from drgrpo_grader import r1_zero_reward_fn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from torch.utils.data import Dataset, DataLoader
from unittest.mock import patch
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from tqdm import tqdm
from pathlib import Path

import os
import json
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

def init_wandb(config):
    wandb.init(project="cs336-assignment5", name=config["exp_id"], config=config)
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")

    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

def load_eval_data(data_path):
    with open("prompts/r1_zero.prompt") as f:
        prompt_template = f.read()
    prompts = []
    ground_truths = []
    with open(data_path, "r") as f:
        for line in f:
            sample = json.loads(line)
            prompts.append(prompt_template.format(question=sample["problem"]))
            ground_truths.append(sample["solution"])
    return {
        "prompts": prompts[:10],
        "ground_truths": ground_truths[:10],
    }


def evaluate(config, policy, vllm_model, eval_data, eval_step):
    load_policy_into_vllm_instance(policy, vllm_model)
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=2048,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    results = evaluate_vllm(
        vllm_model,
        r1_zero_reward_fn,
        eval_data["prompts"],
        eval_data["ground_truths"],
        sampling_params,
        Path("eval_outputs") / config["exp_id"] / f"step_{eval_step}.jsonl",
    )
    wandb.log({
        "eval_step": eval_step,
        "eval/format_reward": results["format_reward"],
        "eval/answer_reward": results["answer_reward"],
        "eval/reward": results["reward"],
    })


def train(config):
    init_wandb(config)
    vllm_model = init_vllm(
        config["model_id"],
        config["inference_device"],
        config["seed"],
        0.65,
    )
    print(f"vllm_model.device={vllm_model.llm_engine.model_executor.driver_worker.device}")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(config["train_device"])
    print(f"model.device={model.device}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])

    dataset = SFTDataset(config["train_data"])
    dataloader = DataLoader(dataset, batch_size=config["micro_batch_size"], shuffle=True)
    eval_data = load_eval_data(config["eval_data"])

    train_step = 0
    train_loss = 0.
    #evaluate(config, model, vllm_model, eval_data, train_step)
    for epoch in range(config["n_epochs"]):
        for batch_id, batch in tqdm(enumerate(dataloader)):
            prompts, responses = batch
            batch = tokenize_prompt_and_output(prompts, responses, tokenizer)
            batch = {k:v.to(config["train_device"]) for k,v in batch.items()}

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

            if (batch_id+1) % config["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                train_step += 1
                wandb.log({
                    "train_step": train_step,
                    "train/loss": train_loss,
                })
                train_loss = 0.

                if train_step % config["eval_steps"] == 0:
                    evaluate(config, model, vllm_model, eval_data, train_step)

if __name__ == "__main__":
    config = {
        "exp_id": "sft-1",
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
        "eval_steps": 100,
        "n_epochs": 1,
    }
    train(config)