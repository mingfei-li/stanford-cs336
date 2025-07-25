import time
import os
import regex as re
from collections import Counter
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from typing import BinaryIO

PRETOKENIZER_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

"""
Helper function that finds the first split_special_token 
at or after a strat_pos.
"""
def find_boundary_in_chunk(
    input_path: str | os.PathLike, 
    split_special_token: bytes,
    start_pos: int,
) -> int:
    mini_chunk_size = 4096
    with open(input_path, "rb") as file:
        file.seek(start_pos)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                return start_pos + found_at
            start_pos += mini_chunk_size

        # If no boundary is found, return file size
        return file.tell()

def find_chunk_boundaries(
    input_path: str | os.PathLike, 
    chunk_size: int, 
    split_special_token: bytes,
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

    desired_num_chunks = file_size // chunk_size
    if file_size % chunk_size != 0:
        desired_num_chunks += 1

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    #with Pool() as p:
    chunk_boundaries[1:-1] = map(
        partial(find_boundary_in_chunk, input_path, split_special_token),
        chunk_boundaries[1:-1],
    )

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pretokenize_chunk(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    boundaries: tuple[int, int],
) -> Counter:
    start, end = boundaries
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8")
    escaped_special_tokens = [token.replace("|", r"\|") for token in special_tokens]
    documents = re.split("|".join(escaped_special_tokens), chunk)
    pretoken_counts = Counter(
        tuple(bytes([byte]) for byte in pretoken.encode("utf-8"))
        for doc in documents
        for pretoken in re.findall(PRETOKENIZER_PAT, doc)
    )
    return pretoken_counts

def count_token_pairs(
    pretoken_counts: tuple[bytes],
) -> dict[tuple[bytes, bytes], int]:
    pretoken, multiplier = pretoken_counts
    pair_counts = defaultdict(int)
    for token_1, token_2 in zip(pretoken[:-1], pretoken[1:]):
       pair_counts[(token_1, token_2)] += multiplier
    return pair_counts

def apply_merge(
    merge: tuple[bytes, bytes],
    pretoken,
) -> Counter[tuple[bytes]]:
    new_pretoken = []
    i = 0
    merged_token = merge[0] + merge[1]
    n = len(pretoken)
    while i < n:
        if i+1 < n and pretoken[i] == merge[0] and pretoken[i+1] == merge[1]:
            new_pretoken.append(merged_token)
            i += 2
        else:
            new_pretoken.append(pretoken[i])
            i += 1
    return tuple(new_pretoken)

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
     with Pool() as p:
        boundaries = find_chunk_boundaries(
            input_path, 1_000_000, "<|endoftext|>".encode("utf-8"))

        pretoken_count_dict = sum(p.map(
            partial(pretokenize_chunk, input_path, special_tokens),
            list(zip(boundaries[:-1], boundaries[1:])),
        ), Counter())
        pretokens = pretoken_count_dict.keys()
        pretoken_counts = pretoken_count_dict.values()

        token_set = (
            set(token for pretoken in pretokens for token in pretoken)
            | set(token.encode("utf-8") for token in special_tokens + ["<|endoftext|>"])
        )
        merges = []

        while len(token_set) < vocab_size:
            pair_counts_per_pretoken = p.map(
                count_token_pairs,
                list(zip(pretokens, pretoken_counts)),
            )
            total_pair_counts = defaultdict(int)
            for pair_counts in pair_counts_per_pretoken:
                for pair, count in pair_counts.items():
                    total_pair_counts[pair] += count

            merge, _ = max(total_pair_counts.items(), key=lambda x: (x[1], x[0]))
            merges.append(merge)
            token_set.add(merge[0]+merge[1])
            pretokens = list(p.map(
                partial(apply_merge, merge),
                pretokens,
            ))

        vocab = {index: token for index, token in enumerate(token_set)}
        return vocab, merges

if __name__ == "__main__":
    print(os.getcwd())
    start_time = time.time()
    vocab, merges = train_bpe(
        "data/TinyStoriesV2-GPT4-valid.txt",
        10000,
        ["<|endoftext|>"],
    )
    end_time = time.time()
    print(f"Total time: {end_time - start_time} seconds")