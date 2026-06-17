from datasets import load_dataset
from tqdm import tqdm

LANG = "cs"
for split in ["train", "validation"]:
    dataset = load_dataset("wmt19", f"{LANG}-en", split=split, streaming=True)
    with open(
        f"/mnt/storage/hopkins/data/wmt19/{split}.{LANG}-en.{LANG}", "w"
    ) as foreign:
        with open(f"/mnt/storage/hopkins/data/wmt19/{split}.{LANG}-en.en", "w") as en:
            for line in tqdm(dataset):
                foreign.write(line["translation"][LANG] + "\n")
                en.write(line["translation"]["en"] + "\n")
