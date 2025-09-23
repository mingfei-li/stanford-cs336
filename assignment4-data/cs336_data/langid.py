import argparse
import fasttext
import gzip
import random
import regex as re

from cs336_data.extract import extract_text_from_html_bytes
from fastwarc.warc import ArchiveIterator, WarcRecordType
from tqdm import tqdm


MODEL_PATH = "data/lid.176.bin"

def identify_language(text: str) -> tuple[any, float]:
    text = text.replace("\n", " ")
    model = fasttext.load_model(MODEL_PATH)
    labels, probs = model.predict(text)
    return labels[0].replace("__label__", ""), probs[0]

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
                    sample_id += 1

                    html_bytes = record.reader.read()
                    text = extract_text_from_html_bytes(html_bytes)
                    language, prob = identify_language(text)
                    
                    parts = re.split(r"\s+", text)

                    f_out.write(f"\n===== Sample {sample_id} =====\n")
                    f_out.write(f"Language: {language}\n")
                    f_out.write(f"Probability: {prob:.2%}\n")
                    f_out.write(f"{' '.join(parts)}\n")
                    if sample_id == args.n_samples:
                        break