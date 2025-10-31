import torch
from transformers import PreTrainedTokenizerBase

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    prompt_ids = tokenizer(prompt_strs)["input_ids"]
    output_ids = tokenizer(output_strs)["input_ids"]
    all_ids = {
        "input_ids": [prompt + output for prompt, output in zip(prompt_ids, output_ids)]
    }
    tokens = tokenizer.pad(all_ids, return_tensors="pt")["input_ids"]
    input_ids = tokens[:, :-1]
    labels = tokens[:, 1:]
    response_mask = torch.zeros_like(labels)
    for i in range(labels.shape[0]):
        response_mask[i, len(prompt_ids[i])-1:len(prompt_ids[i])+len(output_ids[i])-1] = 1
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }