import json
import os
import regex as re
import sys
import time
from collections import Counter
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
from typing import BinaryIO

from tests.common import gpt2_bytes_to_unicode

PRETOKENIZER_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT2_BYTES_TO_UNICODE = gpt2_bytes_to_unicode()
N_CHUNKS = 100

def printable(token: bytes):
    return "".join(GPT2_BYTES_TO_UNICODE[byte] for byte in token)

def find_chunk_boundaries(
    input_path: str | os.PathLike,
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    with open(input_path, "rb") as file:
        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

def pretokenize_chunk(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    boundaries: tuple[int, int],
) -> dict[str, int]:
    start, end = boundaries
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8")
    escaped_special_tokens = [token.replace("|", r"\|") for token in special_tokens]
    documents = re.split("|".join(escaped_special_tokens), chunk)
    pretoken_counts = defaultdict(int)
    for doc in documents:
        for match in re.finditer(PRETOKENIZER_PAT, doc):
            pretoken_counts[match.group()] += 1
    return pretoken_counts

def pretokenize(
    input_path: str | os.PathLike,
    special_tokens: list[str],
) -> tuple[list[tuple[bytes]], list[int]]:
    with Pool() as p:
        boundaries = find_chunk_boundaries(
            input_path, N_CHUNKS, "<|endoftext|>".encode("utf-8"))
    
        pretoken_counts_by_chunk = p.map(
            partial(pretokenize_chunk, input_path, special_tokens),
            list(zip(boundaries[:-1], boundaries[1:])),
        )
        pretoken_counts = defaultdict(int)
        for chunk_pretoken_counts in pretoken_counts_by_chunk:
            for pretoken, count in chunk_pretoken_counts.items():
                pretoken_counts[pretoken] += count
        
        pretokens = [
            tuple(bytes([byte]) for byte in pretoken.encode("utf-8"))
            for pretoken in pretoken_counts
        ]
        return pretokens, pretoken_counts.values()

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    start_time = time.time()
    pretokens, pretoken_counts = pretokenize(input_path, special_tokens)
    pair_counts = defaultdict(int)
    for pretoken, count in zip(pretokens, pretoken_counts):
        for token_1, token_2 in zip(pretoken[:-1], pretoken[1:]):
            pair_counts[(token_1, token_2)] += count

    token_set = set(
        [token.encode("utf-8") for token in special_tokens]
        + [bytes([byte]) for byte in range(256)]
    )
    merges = []
    after_pretokenization = time.time()
    print(f"Pretokenization: {after_pretokenization - start_time}")

    for _ in tqdm(range(vocab_size-len(token_set))):
        merge, _ = max(pair_counts.items(), key=lambda x: (x[1], x[0]))
        merged_token = merge[0]+merge[1]
        merges.append(merge)
        token_set.add(merged_token)

        new_pretokens = []
        for pretoken, count in zip(pretokens, pretoken_counts):
            new_pretoken = []
            i = 0
            while i < len(pretoken):
                if i+1 < len(pretoken) and pretoken[i] == merge[0] and pretoken[i+1] == merge[1]:
                    new_pretoken.append(merged_token)
                    i += 2
                else:
                    new_pretoken.append(pretoken[i])
                    i += 1
            for token_1, token_2 in zip(new_pretoken[:-1], new_pretoken[1:]):
                if token_1 == merged_token or token_2 == merged_token:
                    token_1_to_rm = merge[1] if token_1 == merged_token else token_1
                    token_2_to_rm = merge[0] if token_2 == merged_token else token_2
                    pair_counts[(token_1_to_rm, token_2_to_rm)] -= count
                    pair_counts[(token_1, token_2)] += count
            new_pretokens.append(tuple(new_pretoken))
        pretokens = new_pretokens
        del pair_counts[merge]

    after_tokenization = time.time()
    print(f"Tokenization: {after_tokenization - after_pretokenization}")
    vocab = {index: token for index, token in enumerate(token_set)}
    return vocab, merges

if __name__ == "__main__":
    start_time = time.time()
    vocab, merges = train_bpe(
        "data/TinyStoriesV2-GPT4-train.txt",
        10_000,
        ["<|endoftext|>"],
    )

    vocab_out = {printable(token): index for index, token in vocab.items()}
    with open("tokenizer-data/TinyStoriesV2-GPT4-train-vocab.json", "w") as vocab_file:
        json.dump(vocab_out, vocab_file, indent=4, ensure_ascii=False)
    
    with open("tokenizer-data/TinyStoriesV2-GPT4-train-merges.txt", "w") as merge_file:
        for token_1, token_2 in merges:
            merge_file.write(f"{printable(token_1)} {printable(token_2)}\n")

    end_time = time.time()
    print(f"Total time: {end_time - start_time} seconds")