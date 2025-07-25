import os
from typing import BinaryIO

def find_chunk_boundaries(
    input_path: str | os.PathLike, 
    chunk_size: int, 
    split_special_token: bytes,
) -> list[int]:

    """
    Helper function that finds the first split_special_token 
    at or after a strat_pos.
    """
    def find_boundary_in_chunk(
        start_pos: int,
    ) -> int:
        mini_chunk_size = 4096
        with open(input_path, "rb") as file:
            file.seek(start_pos)
            while True:
                mini_chunk = file.read(mini_chunk_size)
                if mini_chunk == b"":
                    break
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    return start_pos + found_at
                start_pos += mini_chunk_size

            # If no boundary is found, return file size
            return file.tell()

    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    with open(input_path, "rb") as file:
        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()

    desired_num_chunks = file_size // chunk_size

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    chunk_boundaries[1:-1] = map(find_boundary_in_chunk, chunk_boundaries[1:-1])

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            input_path, 1_000_000, "<|endoftext|>".encode("utf-8"))
            
        print(boundaries)
        # The following is a serial implementation, but you can parallelize this 
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token

print(os.getcwd())
train_bpe(
    "assignment1-basics-main/data/TinyStoriesV2-GPT4-valid.txt",
    100,
    [],
)