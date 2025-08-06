import json
import regex as re
from .common import gpt2_bytes_to_unicode, PRETOKENIZER_PAT

unicode_to_byte = {v:k for k,v in gpt2_bytes_to_unicode().items()}
def unicode_string_to_bytes(s: str) -> bytes:
    return bytes([unicode_to_byte(c) for c in s])

class Tokenizer():
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] = None,
    ) -> None:
        self.id_to_token = vocab
        self.token_to_id = {v:k for k,v in vocab}
        self.merges = merges
        self.special_tokens = set(special_tokens)
        self.pretoken_encoding_cache = {}

        next_token_id = len(vocab)
        if special_tokens is not None:
            for token in special_tokens:
                token = token.encode("utf-8")
                if token not in self.token_to_id:
                    self.token_to_id[token] = next_token_id
                    self.id_to_token[next_token_id] = token
                    next_token_id += 1

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
                merge = tuple(unicode_string_to_bytes(s) for s in line.split(" "))
                assert len(token_pair) == 2
                merges.append(merge)
        
        return Tokenzier(vocab, merges, special_tokens)

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        if pretoken in self.pretoken_encoding_cache:
            return self.pretoken_encoding_cache[pretoken]
        
        pretoken_bytes = [bytes([byte]) for byte in pretoken.encode("utf-8")]
        n = len(pretoken_bytes)
        for merge in self.merges:
            token_1, token_2 = merge
            merged_token = token_1 + token_2

            i = 0
            j = 0
            while i < n:
                if i+1 < n and pretoken_bytes[i] == token_1 and pretoken_bytes[i+1] == token_2:
                    pretoken_bytes[j] = merged_token
                    j += 1
                    i += 2
                else:
                    pretoken_bytes[j] = pretoken_bytes[i]
                    j += 1
                    i += 1
            n = j
        encoded = [self.token_to_id[token[i]] for i in range(n)]
        self.pretoken_encoding_cache[pretoken] = encoded
        return encoded

    def encode(self, text: str) -> list[int]:
        documents = re.split(
            "(" + "|".join(map(re.escape, self.special_tokens)) + ")",
            text,
        )
        encoded = []
        for doc in documents:
            if doc in self.special_tokens:
                encoded.append(self.token_to_id[doc])
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