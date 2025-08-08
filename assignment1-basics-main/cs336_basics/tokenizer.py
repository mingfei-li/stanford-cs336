from __future__ import annotations
import argparse
import json
import numpy as np
import os
import regex as re
from collections import defaultdict
from time import time
from tqdm import tqdm
from .common import find_chunk_boundaries, gpt2_bytes_to_unicode, PRETOKENIZER_PAT

unicode_to_byte = {v:k for k,v in gpt2_bytes_to_unicode().items()}
def unicode_string_to_bytes(s: str) -> bytes:
    return bytes([unicode_to_byte[c] for c in s])

class Tokenizer():
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] = None,
    ) -> None:
        self.id_to_token = vocab
        self.token_to_id = {v:k for k,v in vocab.items()}
        self.merges = merges
        self.special_tokens = (
            sorted(special_tokens, key=len, reverse=True)
            if special_tokens is not None else None
        )
        self.special_token_set = set(special_tokens or [])
        self.pretoken_encoding_cache = {}

        next_token_id = len(vocab)
        if special_tokens is not None:
            for token in special_tokens:
                token = token.encode("utf-8")
                if token not in self.token_to_id:
                    self.token_to_id[token] = next_token_id
                    self.id_to_token[next_token_id] = token
                    next_token_id += 1

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str],
    ) -> Tokenizer:
        with open(vocab_filepath, "r") as f:
            vocab_vk = json.load(f)
            vocab = {k: unicode_string_to_bytes(v) for v, k in vocab_vk.items()}
        with open(merges_filepath, "r") as f:
            merges = []
            for line in f:
                merge = tuple(unicode_string_to_bytes(s) for s in line.strip().split(" "))
                assert len(merge) == 2
                merges.append(merge)
        
        return cls(vocab, merges, special_tokens)

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        if pretoken in self.pretoken_encoding_cache:
            return self.pretoken_encoding_cache[pretoken]
        
        tokens = [bytes([byte]) for byte in pretoken.encode("utf-8")]
        n = len(tokens)
        index = defaultdict(set)
        for i in range(n-1):
            index[(tokens[i],tokens[i+1])].add(i)
        prev_pos = list(range(-1, n-1))
        next_pos = list(range(1, n+1))

        for merge in self.merges:
            if merge not in index:
                continue

            token_1, token_2 = merge
            merged_token = token_1 + token_2
            for pos_1 in index[merge].copy():
                pos_2 = pos_1 + len(token_1)
                if tokens[pos_1] != token_1 or tokens[pos_2] != token_2:
                    continue
                
                tokens[pos_1] = merged_token
                tokens[pos_2] = None
                next_pos[pos_1] = next_pos[pos_2]
                if next_pos[pos_1] < n:
                    prev_pos[next_pos[pos_1]] = pos_1
                    index[(token_2, tokens[next_pos[pos_1]])].remove(pos_2)
                    index[(merged_token, tokens[next_pos[pos_1]])].add(pos_1)
                if prev_pos[pos_1] >= 0:
                    index[(tokens[prev_pos[pos_1]], token_1)].remove(prev_pos[pos_1])
                    index[(tokens[prev_pos[pos_1]], merged_token)].add(prev_pos[pos_1])
            del index[merge]

        encoded = [self.token_to_id[token] for token in tokens if token is not None]
        self.pretoken_encoding_cache[pretoken] = encoded
        return encoded

    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            documents = re.split(
                "(" + "|".join(map(re.escape, self.special_tokens)) + ")",
                text,
            )
        else:
            documents = [text]
        encoded = []
        for doc in documents:
            if doc in self.special_token_set:
                encoded.append(self.token_to_id[doc.encode("utf-8")])
            else:
                for match in re.finditer(PRETOKENIZER_PAT, doc):
                    pretoken = match.group()
                    encoded.extend(self._encode_pretoken(pretoken))
        return encoded

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            encoded = self.encode(text)
            for token_id in encoded:
                yield token_id

    def decode(self, ids: list[int]) -> str:
        utf8_encoded = b"".join(self.id_to_token[id] for id in ids)
        return utf8_encoded.decode("utf-8", errors="replace")

def sample_text(
    input_filepath: str,
    sample_filepath: str,
    n_samples: int,
    split_special_token: str,
) -> None:
    with open(input_filepath, "rb") as f:
        chunk = f.read(1_000_000).decode("utf-8")
    docs = re.split(re.escape(split_special_token), chunk)
    with open(sample_filepath, "w") as f:
        f.write(split_special_token.join(docs[:n_samples]))

def generate_chunks(
    input_filepath: str | os.PathLike,
    split_special_token: str,
) -> Iterator[str]:
    chunk_boundaries = find_chunk_boundaries(
        input_filepath,
        split_special_token.encode("utf-8"),
        desired_num_chunks=100,
    )
    with open(input_filepath, "rb") as f:
        for start, end in tqdm(
            zip(chunk_boundaries[:-1], chunk_boundaries[1:]),
            total=len(chunk_boundaries)-1,
            desc="Tokenizing",
        ):
            f.seek(start)
            yield f.read(end - start).decode("utf-8")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", help="choose an action from encode/decode")
    parser.add_argument("--tokenizer")
    parser.add_argument("--input")
    parser.add_argument("--n_samples", type=int, default=0)

    args = parser.parse_args()

    vocab_filepath = f"tokenizer_data/{args.tokenizer}.json"
    merges_filepath = f"tokenizer_data/{args.tokenizer}-merges.txt"
    text_filepath = f"data/{args.input}.txt"
    text_sample_filepath = f"data/{args.input}-sample.txt"
    decoded_text_filepath = f"data/{args.input}-decoded.txt"
    token_ids_filepath = f"data/{args.input}-token-ids.bin"
    split_special_token = "<|endoftext|>"

    tokenizer = Tokenizer.from_files(
        vocab_filepath,
        merges_filepath,
        [split_special_token],
    )

    start_time = time()
    if args.action == "encode":
        if args.n_samples > 0:
            sample_text(
                text_filepath,
                text_sample_filepath,
                args.n_samples,
                split_special_token,
            )
            text_filepath = text_sample_filepath
        with open(token_ids_filepath, "wb") as f:
            chunk_generator = tokenizer.encode_iterable(
                generate_chunks(
                    text_filepath,
                    split_special_token,
                )
            )
            token_ids = np.fromiter(chunk_generator, dtype=np.uint16)
            f.write(token_ids.tobytes())
    elif args.action == "decode":
        token_ids = np.fromfile(token_ids_filepath, dtype=np.uint16)
        decoded_text = tokenizer.decode(token_ids)
        with open(decoded_text_filepath, "w") as f:
            f.write(decoded_text)
    end_time = time()
    print(f"Total time: {end_time - start_time:.2f}")