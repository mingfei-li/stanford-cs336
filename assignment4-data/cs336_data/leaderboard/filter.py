import argparse
import os
from multiprocessing import cpu_count

import gzip
import submitit
from fastwarc.warc import ArchiveIterator
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

def process_single_wet_file(
    input_file: os.PathLike,
    output_dir: os.PathLike,
    discarded_dir: os.PathLike
):
    def discard(text: str, reason: str):
        if reason not in discarded:
            discarded.add(reason)
            with open(os.path.join(discarded_dir, f"{reason}.txt"), "w") as f:
                f.write(text)

    count = 0
    discarded = set()
    filename = os.path.basename(input_file)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(discarded_dir, exist_ok=True)

    count = 0
    with gzip.open(input_file, "rb") as f_in:
        for record in ArchiveIterator(f_in):
            count += 1
    return input_file
    
    with gzip.open(input_file, "rb") as f_in:
        for record in tqdm(ArchiveIterator(f_in), desc=filename, total=count):
            text = record.reader.read().decode("utf-8")
            if not language_filter(text):
                discard(text, "lanugage")
                continue
            if not gopher_quality_filter(text):
                discard(text, "gopher")
                continue
            if not toxicty_filter(text):
                discard(text, "toxicity")
                continue
            if not quality_classifier_filter(text):
                discard(text, "quality_classifier")
                continue
            text = redact_pii(text)

            assert text
            out_path = os.path.join(output_dir, f"out_{count}.txt")
            with open(out_path, "w") as f_out:
                f_out.write(text)
            count += 1
    return input_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir")
    parser.add_argument("--slurm_partition")
    parser.add_argument("--batch_size", type=int)
    args = parser.parse_args()

    executor = submitit.AutoExecutor(folder="data/slurm_logs")
    executor.update_parameters(
        slurm_partition=args.slurm_partition,
        cpus_per_task=1,
        timeout_min=60*24*3,
    )

    input_dir = os.path.join(args.base_dir, "input")
    wet_files = os.listdir(input_dir)
    for batch_start in range(0, len(wet_files), args.batch_size):
        jobs = []
        with executor.batch():
            for wet_file in wet_files[batch_start:batch_start+args.batch_size]:
                input_file = os.path.join(input_dir, wet_file)
                output_dir = os.path.join(args.base_dir, f"output/{wet_file}")
                discard_dir = os.path.join(args.base_dir, f"discard/{wet_file}")
                job = executor.submit(
                    process_single_wet_file,
                    input_file,
                    output_dir,
                    discard_dir,
                )
                jobs.append(job)

        print(f"Batch submitted: start_index={batch_start}, batch_size={len(jobs)}")
        for job in tqdm(submitit.helpers.as_completed(jobs), total=len(jobs)):
            print(f"Processing complete: {job.result()}")

if __name__ == "__main__":
    main()
