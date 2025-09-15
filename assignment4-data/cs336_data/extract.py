import argparse
import gzip

from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
from tqdm import tqdm

def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    encoding = detect_encoding(html_bytes)
    decoded = html_bytes.decode(encoding, errors="replace")
    return extract_plain_text(decoded)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")

    args = parser.parse_args()
    with gzip.open(args.input, "rb") as fin, gzip.open(args.output, "wb") as fout:
        for record in tqdm(ArchiveIterator(fin)):
            if record.record_type == WarcRecordType.response:
                html_bytes = record.reader.read()
                text = extract_text_from_html_bytes(html_bytes)
                fout.write(text.encode("utf-8"))