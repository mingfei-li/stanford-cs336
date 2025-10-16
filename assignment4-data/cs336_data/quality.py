import argparse
import gzip
import nltk
import numpy as np
import random
import regex as re

from cs336_data.extract import extract_text_from_html_bytes
from fastwarc.warc import ArchiveIterator, WarcRecordType
from tqdm import tqdm
from typing import Any

def gopher_quality_filter(text: str) -> bool:
    words = nltk.word_tokenize(text)
    if len(words) < 50 or len(words) > 100_000:
        return False

    mean_word_len = np.mean([len(word) for word in words])
    if mean_word_len < 3 or mean_word_len > 10:
        return False
    
    lines = text.split("\n")
    problematic_lines = [line for line in lines if line.endswith("...")]
    if len(problematic_lines) > 0.3 * len(lines):
        return False

    good_words = [
        word for word in words 
        if any(ch.isalpha() for ch in word)
    ]
    if len(good_words) < 0.8 * len(words):
        return False

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--n_samples", type=int)
    args = parser.parse_args()

    sample_id = 0
    with gzip.open(args.input, "rb") as f_in, open(args.output, "w") as f_out:
        for record in tqdm(ArchiveIterator(f_in)):
            if record.record_type == WarcRecordType.response:
                if random.random() < 0.01:
                    html_bytes = record.reader.read()
                    text = extract_text_from_html_bytes(html_bytes)
                    pass_gopher = gopher_quality_filter(text)

                    sample_id += 1
                    f_out.write(f"\n===== Sample {sample_id} =====\n")
                    f_out.write(f"Pass gopher: {pass_gopher}\n")
                    f_out.write(f'{re.sub(r"\s+", " ", text)}\n')
                    if sample_id == args.n_samples:
                        break