import argparse
import fasttext
import gzip
import random
import regex as re

from cs336_data.extract import extract_text_from_html_bytes
from fastwarc.warc import ArchiveIterator, WarcRecordType
from tqdm import tqdm

EMAIL_PATTERN = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
PHONE_NUMBER_PATTERN = r"""
    (?:(?:\+?1\s*(?:[.-]\s*)?)?          # optional country code
    (?:\(\s*\d{3}\s*\)|\d{3})            # area code with or without parentheses
    (?:\s*[.-]?\s*)                      # separator
    \d{3}                                # first 3 digits
    (?:\s*[.-]?\s*)                      # separator
    \d{4})                               # last 4 digits
"""
IP_PATTERN = r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}" \
    r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"

def mask_emails(text: str) -> tuple[str, int]:
    return re.subn(EMAIL_PATTERN, "|||EMAIL_ADDRESS|||", text)

def mask_phone_numbers(text: str) -> tuple[str, int]:
    return re.subn(PHONE_NUMBER_PATTERN, "|||PHONE_NUMBER|||", text, flags=re.VERBOSE)

def mask_ips(text: str) -> tuple[str, int]:
    return re.subn(IP_PATTERN, "|||IP_ADDRESS|||", text)


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
                html_bytes = record.reader.read()
                text_orig = extract_text_from_html_bytes(html_bytes)

                text, n_emails = mask_emails(text_orig)
                text, n_phone_numbers = mask_phone_numbers(text)
                text, n_ips = mask_ips(text)
                
                n_masks = n_emails + n_phone_numbers + n_ips

                if n_masks > 0 and random.random() < 0.01:
                    sample_id += 1
                    f_out.write(f"\n===== Sample {sample_id} =====\n")
                    f_out.write(f"Emails masked: {n_emails}\n")
                    f_out.write(f"Phone numbers masked: {n_phone_numbers}\n")
                    f_out.write(f"IP addresses masked: {n_ips}\n")
                    f_out.write(f'O: {re.sub(r"\s+", " ", text_orig)}\n')
                    f_out.write(f'M: {re.sub(r"\s+", " ", text)}\n')
                    if sample_id == args.n_samples:
                        break