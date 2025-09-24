import hashlib
import os
import random
import shutil
import unicodedata

from collections import defaultdict

def exact_line_deduplication(
    input_files: list[os.PathLike],
    output_directory: os.PathLike,
) -> None:
    line_counts = defaultdict(int)
    for input_file in input_files:
        with open(input_file, "r") as f_in:
            for line in f_in:
                sha256 = hashlib.sha256(line.encode("utf-8")).hexdigest()
                line_counts[sha256] += 1
    
    for input_file in input_files:
        output_file = os.path.join(
            output_directory,
            os.path.basename(input_file),
        )
        with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
            for line in f_in:
                sha256 = hashlib.sha256(line.encode("utf-8")).hexdigest()
                if line_counts[sha256] == 1:
                    f_out.write(line)

def normalize(text: str) -> str:
    # nfd normalization
    text = unicodedata.normalize("NFD", text)
    text = text.casefold()              # to lower case

    filtered_chars = []
    for ch in text:
        category = unicodedata.category(ch)
        if category == "Mn":            # remove accents
            continue
        if category.startswith("P"):    # remove punctuations
            continue
        filtered_chars.append(ch)
    text = "".join(filtered_chars)

    text = " ".join(text.split())       # collapse whitespace
    return text

def get_shingles(file: os.PathLike, ngrams: int) -> list[bytes]:
    with open(file, "r") as f:
        text = f.read()
    encoded = normalize(text).encode("utf-8")
    shingles = [encoded[i:i+ngrams] for i in range(len(encoded)-ngrams+1)]
    return shingles

def minhash_signature(
    file: os.PathLike,
    seeds: list[bytes],
    ngrams: int,
) -> tuple[int]:
    shingles = get_shingles(file, ngrams)
    signature = []
    for seed in seeds:
        min_digest = 2**64-1
        for s in shingles:
            digest = hashlib.blake2b(s, digest_size=8, key=seed).digest()
            min_digest = min(min_digest, int.from_bytes(digest))
        signature.append(min_digest)
    return tuple(signature)
                
class LSHIndex():
    def __init__(self, num_bands: int):
        self.band_index = [defaultdict(set) for _ in range(num_bands)]
    
    def add(self, minhash: tuple[int], value: str):
        band_size = len(minhash) // len(self.band_index)
        for index, i in zip(self.band_index, range(0, len(minhash), band_size)):
            index[minhash[i:i+band_size]].add(value)
    
    def get(self, minhash: tuple[str]) -> set[str]:
        result = set()
        band_size = len(minhash) // len(self.band_index)
        for index, i in zip(self.band_index, range(0, len(minhash), band_size)):
            result = result | index[minhash[i:i+band_size]]
        return result

class UnionFind():
    def __init__(self, elements: list[str]):
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}
    
    def find(self, a: str) -> str:
        if self.parent[a] != a:
            self.parent[a] = self.find(self.parent[a])
        return self.parent[a]
    
    def union(self, a: str, b: str):
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        elif self.rank[root_b] > self.rank[root_a]:
            self.parent[root_a] = root_b
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1

def jaccard_similarity(
    file_1: os.PathLike,
    file_2: os.PathLike,
    ngrams: int,
) -> float:
    shingles1 = set(get_shingles(file_1, ngrams))
    shingles2 = set(get_shingles(file_2, ngrams))
    intersect = shingles1 & shingles2
    union = shingles1 | shingles2
    return len(intersect) / len(union)
    
def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    seeds = [random.randint(0, 2**64-1).to_bytes(8) for _ in range(num_hashes)]
    signatures = [minhash_signature(file, seeds, ngrams) for file in input_files]

    lsh_index = LSHIndex(num_bands)
    for file, signature in zip(input_files, signatures):
        lsh_index.add(signature, file)

    uf_set = UnionFind(input_files)
    for file, signature in zip(input_files, signatures):
        for target in lsh_index.get(signature):
            if jaccard_similarity(file, target, ngrams) > jaccard_threshold:
                uf_set.union(file, target)
    
    for file in input_files:
        if uf_set.find(file) == file:
            shutil.copy(file, output_directory)