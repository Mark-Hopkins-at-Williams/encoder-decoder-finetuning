import os
from pathlib import Path
from random import shuffle
from tqdm import tqdm
from transformers import AutoTokenizer


def reorganize(batch_size, root_dir, split, output_dir):
    """
    Reorganizes text files by sorting lines by token length and shuffling in batches.

    This function tokenizes each line of the English split file using a pretrained tokenizer,
    sorts all lines by their tokenized length, chunks them into batches, shuffles the batch
    order, and then reorders all split-related files accordingly. The reorganized files are
    saved to `output_dir`.

    Parameters
    ----------
    batch_size : int
        Number of lines per shuffled chunk. Shuffling is done at the chunk level.
    root_dir : Path
        Path to the directory containing input files named like `<split>.*`.
    split : str
        Prefix of the files to process (e.g., "train" for "train.en", "train.fr", etc.).
    output_dir : Path
        Path to the directory where reorganized files will be written. Must not exist prior to call.

    Raises
    ------
    FileExistsError
        If `output_dir` already exists.
    """
    
    os.mkdir(output_dir)
    files = list(root_dir.glob(f"{split}.*"))
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lengths = []
    with open(root_dir / f"{split}.en") as reader:
        for i, line in tqdm(enumerate(reader)):
            line = line.strip()
            tokens = tokenizer(line)["input_ids"]
            lengths.append((len(tokens), i))
    line_nums_by_length = [line_num for _, line_num in sorted(lengths)]
    chunk_starts = [
        batch_size * k for k in range(len(line_nums_by_length) // batch_size)
    ]
    shuffle(chunk_starts)
    line_nums = []
    for start in chunk_starts:
        line_nums.extend(line_nums_by_length[start : start + batch_size])

    for file in files:
        lines = []
        with open(file) as reader:
            for line in tqdm(reader):
                lines.append(line.strip())
        with open(output_dir / file.name, "w") as writer:
            for num in tqdm(line_nums):
                writer.write(lines[num] + "\n")


if __name__ == "__main__":
    reorganize(
        128,
        Path("/mnt/storage/hopkins/data/flores/test/"),
        "test",
        Path("./foo.test"),
    )
