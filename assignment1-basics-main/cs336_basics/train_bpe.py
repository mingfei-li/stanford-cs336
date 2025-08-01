import json
import os
import regex as re
import sys
import time
from collections import Counter
from collections import defaultdict
from functools import partial
from multiprocessing import cpu_count, Pipe, Pool, Process
from multiprocessing.connection import Connection
from sortedcontainers import SortedSet
from tqdm import tqdm
from typing import BinaryIO

from .common import gpt2_bytes_to_unicode

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
) -> tuple[list[list[bytes]], list[int]]:
    with Pool() as p:
        boundaries = find_chunk_boundaries(
            input_path, N_CHUNKS, "<|endoftext|>".encode("utf-8"))
    
        pretoken_counts_by_chunk = tqdm(
            p.imap_unordered(
                partial(pretokenize_chunk, input_path, special_tokens),
                zip(boundaries[:-1], boundaries[1:]),
            ),
            total=len(boundaries)-1,
            desc="Pretokenization"
        )
        pretoken_counts = defaultdict(int)
        for chunk_pretoken_counts in pretoken_counts_by_chunk:
            for pretoken, count in chunk_pretoken_counts.items():
                pretoken_counts[pretoken] += count
        
        pretokens = [
            tuple(bytes([byte]) for byte in pretoken.encode("utf-8"))
            for pretoken in pretoken_counts
        ]
        return pretokens, list(pretoken_counts.values())

def tokenize(
    pretokens:list[list[bytes]],
    counts: list[int],
    conn: Connection,
) -> tuple[bytes, dict[tuple[bytes], int]]:
    pretoken_lens = [len(pretoken) for pretoken in pretokens]
    while True:
        merge = conn.recv()
        if not merge:
            conn.close()
            break
        
        merged_token = merge[0] + merge[1]
        pair_count_deltas = defaultdict(int)
        for k in range(len(pretokens)):
            pretoken = pretokens[k]
            count = counts[k]
            n = pretoken_lens[k]
            i = 0
            j = 0
            while i < n:
                if i+1 < n and pretoken[i] == merge[0] and pretoken[i+1] == merge[1]:
                    pretoken[j] = merged_token
                    i += 2
                    j += 1
                else:
                    pretoken[j] = pretoken[i]
                    i += 1
                    j += 1
            pretoken_lens[k] = j

            for i in range(j-1):
                token_1 = pretoken[i]
                token_2 = pretoken[i+1]
                if token_1 == merged_token or token_2 == merged_token:
                    token_1_to_rm = merge[1] if token_1 == merged_token else token_1
                    token_2_to_rm = merge[0] if token_2 == merged_token else token_2
                    pair_count_deltas[(token_1_to_rm, token_2_to_rm)] -= count
                    pair_count_deltas[(token_1, token_2)] += count

        conn.send(pair_count_deltas)

def build_indexes(
    pretokens: list[tuple[bytes]]
) -> tuple[defaultdict[SortedSet], list[list[int]], list[list[int]]]:
    pair_index = defaultdict(SortedSet)
    prev_token_index = [None] * len(pretokens)
    next_token_index = [None] * len(pretokens)
    for i in range(len(pretokens)):
        for j in range(len(pretokens[i])-1):
            token_1 = pretokens[i][j]
            token_2 = pretokens[i][j+1]
            pair_index[(token_1,token_2)].add((i, j))
        prev_token_index[i] = list(range(-1, len(pretokens[i])-1))
        next_token_index[i] = list(range(1, len(pretokens[i])+1))

    return pair_index, prev_token_index, next_token_index

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    token_set = set(
        [token.encode("utf-8") for token in special_tokens]
        + [bytes([byte]) for byte in range(256)]
    )
    merges = []

    pair_counts = defaultdict(int)
    pretokens, pretoken_counts = pretokenize(input_path, special_tokens)
    for pretoken, count in zip(pretokens, pretoken_counts):
        for token_1, token_2 in zip(pretoken[:-1], pretoken[1:]):
            pair_counts[(token_1, token_2)] += count
    pair_index, prev_token_index, next_token_index = build_indexes(pretokens)
    count_pair_set = SortedSet((count, pair) for pair, count in pair_counts.items())
    for _ in tqdm(range(vocab_size - len(token_set)), "Tokenization"):
        count, merge = count_pair_set[-1]
        token_1, token_2 = merge
        merged_token = token_1 + token_2

        merges.append(merge)
        token_set.add(merged_token)

        pair_count_deltas = defaultdict(int)
        for pretoken_id, pos_1 in pair_index[merge].copy():
            pos_2 = pos_1 + len(token_1)
            if (
                next_token_index[pretoken_id][pos_1] != pos_2
                or prev_token_index[pretoken_id][pos_2] != pos_1
            ):
                continue

            prev_token_pos = prev_token_index[pretoken_id][pos_1]
            next_token_pos = next_token_index[pretoken_id][pos_2]
            next_token_index[pretoken_id][pos_1] = next_token_pos
            prev_token_index[pretoken_id][pos_2] = None
            next_token_index[pretoken_id][pos_2] = None
            if prev_token_pos >= 0:
                prev_token = b"".join(pretokens[pretoken_id][prev_token_pos:pos_1])
                prev_pair = (prev_token, token_1)
                new_pair_1 = (prev_token, merged_token)

                pair_index[prev_pair].remove((pretoken_id, prev_token_pos))
                pair_count_deltas[prev_pair] -= pretoken_counts[pretoken_id]

                pair_index[new_pair_1].add((pretoken_id, prev_token_pos))
                pair_count_deltas[new_pair_1] += pretoken_counts[pretoken_id]
            if next_token_pos < len(pretokens[pretoken_id]):
                next_token_end = next_token_index[pretoken_id][next_token_pos]
                next_token = b"".join(pretokens[pretoken_id][next_token_pos:next_token_end])
                next_pair = (token_2, next_token)
                new_pair_2 = (merged_token, next_token)

                pair_index[next_pair].remove((pretoken_id, pos_2))
                pair_count_deltas[next_pair] -= pretoken_counts[pretoken_id]

                pair_index[new_pair_2].add((pretoken_id, pos_1))
                pair_count_deltas[new_pair_2] += pretoken_counts[pretoken_id]

                prev_token_index[pretoken_id][next_token_pos] = pos_1

        del pair_index[merge]
        count_pair_set.remove((pair_counts[merge], merge))
        del pair_counts[merge]
        for pair, delta in pair_count_deltas.items():
            count_pair_set.discard((pair_counts[pair], pair))
            pair_counts[pair] += delta
            count_pair_set.add((pair_counts[pair], pair))

    vocab = {index: token for index, token in enumerate(token_set)}
    return vocab, merges

if __name__ == "__main__":
    #dataset = "owt_train"
    dataset = "TinyStoriesV2-GPT4-train"
    start_time = time.time()
    vocab, merges = train_bpe(
        f"data/{dataset}.txt",
        10_000,
        ["<|endoftext|>"],
    )

    vocab_out = {printable(token): index for index, token in vocab.items()}
    with open(f"tokenizer-data/{dataset}.json", "w") as vocab_file:
        json.dump(vocab_out, vocab_file, indent=4, ensure_ascii=False)
    
    with open(f"tokenizer-data/{dataset}-merges.txt", "w") as merge_file:
        for token_1, token_2 in merges:
            merge_file.write(f"{printable(token_1)} {printable(token_2)}\n")

    end_time = time.time()
    print(f"Total time: {end_time - start_time} seconds")