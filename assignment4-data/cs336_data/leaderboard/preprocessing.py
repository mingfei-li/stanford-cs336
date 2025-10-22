import argparse
import os
from functools import partial

import gzip
from fastwarc.warc import ArchiveIterator, WarcRecordType
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from cs336_data.extract import extract_text_from_html_bytes
from cs336_data.langid import identify_language
from cs336_data.pii import (
    mask_emails,
    mask_phone_numbers,
    mask_ips,
)
from cs336_data.quality import gopher_quality_filter
from cs336_data.quality_classifier import classify_quality
from cs336_data.toxicity import classify_nsfw, classify_toxic_speech
from cs336_data.deduplication import (
    exact_line_deduplication,
    minhash_deduplication,
)

def language_filter(text: str) -> bool:
    lang, prob = identify_language(text)
    return lang == "en" and prob > 0.5

def redact_pii(text: str) -> str:
    text, _ = mask_emails(text)
    text, _ = mask_phone_numbers(text)
    text, _ = mask_ips(text)
    return text

def toxicty_filter(text: str) -> bool:
    nsfw_label, nsfw_label_prob = classify_nsfw(text)
    if nsfw_label == "nsfw" or nsfw_label_prob < 0.97:
        return False
    toxic_label, toxic_label_prob = classify_toxic_speech(text)
    if toxic_label == "toxic" or toxic_label_prob < 0.97:
        return False
    return True

def quality_classifier_filter(text: str) -> bool:
    label, _ = classify_quality(text)
    return label == "positive"

def filter_file(base_dir: os.PathLike, input_file: os.PathLike):
    count = 0
    filename = os.path.basename(input_file)
    output_dir = os.path.join(base_dir, "filtered")
    os.makedirs(output_dir, exist_ok=True)
    with gzip.open(input_file, "rb") as f_in:
        for record in tqdm(ArchiveIterator(f_in), desc=filename):
            text = record.reader.read().decode("utf-8")
            if not language_filter(text):
                continue
            if not toxicty_filter(text):
                continue
            if not gopher_quality_filter(text):
                continue
            if not quality_classifier_filter(text):
                continue
            text = redact_pii(text)

            assert text
            out_path = os.path.join(output_dir, f"{filename}.out{count}.txt")
            with open(out_path, "w") as f_out:
                f_out.write(text)
            
            count += 1

def filter(base_dir: os.PathLike):
    input_dir = os.path.join(base_dir, "input")
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)] 
    data_files = [f for f in files if os.path.isfile(f) and f.endswith(".wet.gz")]
    with Pool(processes=cpu_count()) as p:
        list(p.imap(partial(filter_file, base_dir), data_files))

def exact_dedupe(base_dir: os.PathLike):
    input_dir = os.path.join(base_dir, "filtered")
    output_dir = os.path.join(base_dir, "exact_deduped")
    os.makedirs(output_dir, exist_ok=True)
    input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
    exact_line_deduplication(input_files, output_dir)

def fuzzy_dedupe(base_dir: os.PathLike):
    input_dir = os.path.join(base_dir, "exact_deduped")
    output_dir = os.path.join(base_dir, "deduped")
    os.makedirs(output_dir, exist_ok=True)
    input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
    minhash_deduplication(
        input_files=input_files,
        num_hashes=500,
        num_bands=50,
        ngrams=5,
        jaccard_threshold=0.8,
        output_directory=output_dir,
    )

def merge(base_dir: os.PathLike):
    input_dir = os.path.join(base_dir, "deduped")
    input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
    output_file = os.path.join(base_dir, "output.txt")
    with open(output_file, "w") as f_out:
        for file in input_files:
            with open(file, "r") as f:
                text = f.read()
            if not text:
                print(f"Empty file: {file}")
            f_out.write(f"{text}")
            f_out.write(f"<|endoftext|>")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd")
    parser.add_argument("--base_dir")

    args = parser.parse_args()

    if args.cmd == "filter":
        filter(args.base_dir)
    elif args.cmd == "exact_dedupe":
        exact_dedupe(args.base_dir)
    elif args.cmd == "fuzzy_dedupe":
        fuzzy_dedupe(args.base_dir)
    elif args.cmd == "merge":
        merge(args.base_dir)

if __name__ == "__main__":
    main()
