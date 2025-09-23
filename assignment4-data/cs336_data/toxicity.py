import argparse
import fasttext
import gzip
import random
import regex as re

from cs336_data.extract import extract_text_from_html_bytes
from fastwarc.warc import ArchiveIterator, WarcRecordType
from tqdm import tqdm
from typing import Any

NSFW_MODEL_PATH = "data/models/jigsaw_fasttext_bigrams_nsfw_final.bin"
TOXIC_SPEECH_MODEL_PATH = "data/models/jigsaw_fasttext_bigrams_hatespeech_final.bin"

def predict(text: str, model_path: str) -> tuple[Any, float]:
    model = fasttext.load_model(model_path)
    labels, probs = model.predict(text.replace("\n", " "))
    return labels[0].replace("__label__", ""), probs[0]

def classify_nsfw(text: str) -> tuple[Any, float]:
    return predict(text, NSFW_MODEL_PATH)

def classify_toxic_speech(text: str) -> tuple[Any, float]:
    return predict(text, TOXIC_SPEECH_MODEL_PATH)

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
                    nsfw, nsfw_prob = classify_nsfw(text)
                    toxic, toxic_prob = classify_toxic_speech(text)

                    sample_id += 1
                    f_out.write(f"\n===== Sample {sample_id} =====\n")
                    f_out.write(f"nsfw: {nsfw}, prob: {nsfw_prob:.2%}\n")
                    f_out.write(f"toxic: {toxic}: {toxic_prob:.2%}\n")
                    f_out.write(f'{re.sub(r"\s+", " ", text)}\n')
                    if sample_id == args.n_samples:
                        break