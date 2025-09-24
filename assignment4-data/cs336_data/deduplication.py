import hashlib
import os

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