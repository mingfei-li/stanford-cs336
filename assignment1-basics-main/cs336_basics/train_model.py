import argparse
import numpy as np
import numpy.typing as npt
import os
import torch
import wandb
from tqdm import tqdm

from cs336_basics.data import get_batch
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.serialization import save_checkpoint

def get_dataset(filepath: str | os.PathLike) -> npt.NDArray:
    filesize = os.path.getsize(filepath)
    dataset_size = filesize // np.dtype(np.uint16).itemsize
    return np.memmap(filepath, dtype=np.uint16, shape=(dataset_size,))

def get_val_loss(model, val_dataset, args):
    x, y = get_batch(
        val_dataset,
        args.batch_size,
        args.context_length,
        args.device,
    )
    with torch.no_grad():
        logits = model(x)
        loss = cross_entropy(logits, y)
    return loss

def train(args: argparse.Namespace):
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.exp,
        config=vars(args)
    )
    os.makedirs(os.path.join(args.checkpoint_path, args.exp), exist_ok=True)

    train_dataset = get_dataset(args.train_dataset)
    val_dataset = get_dataset(args.val_dataset)

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
    optimizer = AdamW(
        model.parameters(),
        args.max_lr,
        args.weight_decay,
        (args.beta1, args.beta2),
    )

    for t in tqdm(range(args.n_iters), "Training step: "): 
        x, y = get_batch(
            train_dataset,
            args.batch_size,
            args.context_length,
            args.device,
        )
        logits = model(x)
        loss = cross_entropy(logits, y)

        lr = get_lr_cosine_schedule(
            t,
            args.max_lr,
            args.min_lr,
            args.warmup_iters,
            args.n_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.gradient_clipping)
        optimizer.step()

        if t % args.log_every_k == 0:
            val_loss = get_val_loss(model, val_dataset, args)
            run.log({"train_loss": loss.item(), "val_loss": val_loss.item(), "lr": lr}, step=t)
        
        if t % args.checkpoint_every_k == 0:
            save_checkpoint(
                model,
                optimizer,
                t,
                os.path.join(args.checkpoint_path, args.exp, f"model-{t}.pt"),
            )

    save_checkpoint(
        model,
        optimizer,
        t,
        os.path.join(args.checkpoint_path, args.exp, f"model-final.pt"),
    )
    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--train_dataset")
    parser.add_argument("--val_dataset")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--train_val_split", type=float)

    # model hyper parameters
    parser.add_argument("--vocab_size", type=int)
    parser.add_argument("--context_length", type=int)
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--d_ff", type=int)
    parser.add_argument("--rope_theta", type=float)
    parser.add_argument("--n_layers", type=int)
    parser.add_argument("--n_heads", type=int)
    parser.add_argument("--device", default="cpu")

    # learning schedule
    parser.add_argument("--n_iters", type=int)
    parser.add_argument("--max_lr", type=float)
    parser.add_argument("--min_lr", type=float)
    parser.add_argument("--warmup_iters", type=int)

    # optimizer hyper parameters
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--beta1", type=float)
    parser.add_argument("--beta2", type=float)
    parser.add_argument("--gradient_clipping", type=float)

    # wandb setting
    parser.add_argument("--wandb_entity")
    parser.add_argument("--wandb_project")
    parser.add_argument("--log_every_k", type=int)
    parser.add_argument("--exp", default=None)
    parser.add_argument("--checkpoint_every_k", type=int)

    # checkpoint
    parser.add_argument("--checkpoint_path")
    
    args = parser.parse_args()
    
    train(args)