import argparse
import gzip
import regex as re

from fastwarc.warc import ArchiveIterator, WarcRecordType
from tqdm import tqdm

from cs336_data.extract import extract_text_from_html_bytes
from cs336_data.quality import gopher_quality_filter
from cs336_data.toxicity import classify_nsfw, classify_toxic_speech

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--positive", action="store_true")
    parser.add_argument("--limit", type=int, default=0)

    args = parser.parse_args()
    n_samples = 0
    with gzip.open(args.input, "rb") as f_in, gzip.open(args.output, "wb") as f_out:
        for record in tqdm(ArchiveIterator(f_in)):
            if record.record_type == WarcRecordType.response:
                html_bytes = record.reader.read()
                text = extract_text_from_html_bytes(html_bytes)
                prefix = "__label__negative "
                if args.positive:
                    if not gopher_quality_filter(text):
                        continue

                    nsfw, probs = classify_nsfw(text)
                    if nsfw == "nsfw" or (nsfw == "non-nsfw" and probs < 0.97):
                        continue

                    toxic, probs = classify_toxic_speech(text)
                    if toxic == "toxic" or (toxic == "non-toxic" and probs < 0.97):
                        continue
                    prefix = "__label__positive "
                
                f_out.write((prefix + re.sub(r"\s+", " ", text) + "\n").encode("utf-8"))
                n_samples += 1
                if n_samples == args.limit:
                    break


