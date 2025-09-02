import argparse
import torch
from collections.abc import Iterable

from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer

def sample(
    logits: torch.Tensor,
    temperature: float | None = None,
) -> int:
    if temperature is not None:
        logits = logits / (temperature + 1e-6)
    dist = torch.distributions.Categorical(logits=logits)
    return dist.sample()

def nucleus_sample(
    logits: torch.Tensor,
    threshold: float,
) -> int:
    values, indices = torch.sort(logits, descending=True)
    p = torch.softmax(values, dim=-1)
    cumsum = torch.cumsum(p, dim=-1)
    mask = cumsum >= threshold
    if mask.any():
        cutoff = torch.argmax(mask, dim=-1)
    else:
        cutoff = mask.shape[0]
    index_in_top = sample(values[:cutoff+1])
    return indices[index_in_top]

def generate_completion(
    model: TransformerLM,
    eos_token: int,
    prompt: list[int],
    max_len: int,
    temperature: float | None = None,
    nucleus_sampling_threshold: float | None = None,
    device: torch.device | None = None,
) -> list[int]:
    assert len(prompt)

    x = torch.tensor(prompt, device=device)
    while x[-1].item() != eos_token and x.shape[0] < max_len:
        logits = model(x)
        if nucleus_sampling_threshold is not None:
            next_token = nucleus_sample(logits[-1], nucleus_sampling_threshold)
        else:
            next_token = sample(logits[-1], temperature)
        x = torch.cat([x, torch.tensor([next_token], device=device)])

    return x.tolist()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_vocab")
    parser.add_argument("--tokenizer_merges")
    parser.add_argument("--special_token")
    parser.add_argument("--prompt")
    parser.add_argument("--model")
    parser.add_argument("--device")
    parser.add_argument("--max_len", type=int)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--nucleus_sampling_threshold", type=float, default=None)
    parser.add_argument("--vocab_size", type=int)
    parser.add_argument("--context_length", type=int)
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--d_ff", type=int)
    parser.add_argument("--rope_theta", type=int)
    parser.add_argument("--n_layers", type=int)
    parser.add_argument("--n_heads", type=int)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(
        args.tokenizer_vocab,
        args.tokenizer_merges,
        [args.special_token],
    )
    model = TransformerLM(
        args.vocab_size,
        args.context_length,
        args.d_model,
        args.n_layers,
        args.n_heads,
        args.d_ff,
        args.rope_theta,
        args.device,
    )
    model.load_state_dict(torch.load(args.model, args.device)["model"])

    completion = generate_completion(
        model,
        tokenizer.encode(args.special_token)[0],
        tokenizer.encode(args.prompt),
        args.max_len,
        args.temperature,
        args.nucleus_sampling_threshold,
        args.device,
    )

    print(tokenizer.decode(completion))
