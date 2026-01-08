from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

import os
import json
import torch

from utils import (
    get_response_log_probs,
    masked_normalize,
    tokenize_prompt_and_output
)

class SFTDataset(Dataset):
    def __init__(self, file_path: os.PathLike):
        with open(file_path, "r") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]["prompt"], self.data[index]["response"]

def train(config):
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(config.train_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    dataset = SFTDataset(config.data_path)
    dataloader = DataLoader(dataset, shuffle=True, collate_fn=collate_fn)

    for epoch in range(config.n_epochs):
        for batch_id, batch in enumerate(dataloader):
            prompts, responses = zip(*batch)
            batch = tokenize_prompt_and_output(prompts, responses, tokenizer)

            results = get_response_log_probs(
                model,
                batch["input_ids"],
                batch["labels"],
                return_token_entropy=True,
            )

            loss, _ = sft_microbatch_train_step(
                results["log_probs"],
                batch["response_mask"],
                config.gradient_accumulation_steps,
            )

            if (batch_id+1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

