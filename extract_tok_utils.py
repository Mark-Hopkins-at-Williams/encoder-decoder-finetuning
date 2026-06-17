from transformers import AutoTokenizer

from collections import defaultdict
import torch
import subprocess
import tempfile
from typing import Dict, List


def parse_fast_align_sentence(
    align_line: str,
    src_tokens: list[str],
    tgt_tokens: list[str],
):
    """
    Returns:
        dict[int, list[int]]  # src_idx -> tgt_idxs
    """
    alignment = defaultdict(list)

    for pair in align_line.strip().split():
        i, j = pair.split("-")
        i, j = int(i), int(j)

        if i < len(src_tokens) and j < len(tgt_tokens):
            alignment[i].append(j)

    return dict(alignment)


def tokenize_parallel_to_fast_align(
    src_file,
    tgt_file,
    out_file,
    src_lang,
    tgt_lang,
    model_name="facebook/nllb-200-distilled-600M",
):
    """
    Tokenizes parallel files using NLLB and writes output in fast_align format:

    src_tok1 src_tok2 ... ||| tgt_tok1 tgt_tok2 ...

    Skips:
    - language tokens (__eng_Latn__)
    - special tokens (<s>, </s>, <pad>, etc.)
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_line(line, lang_code):
        tokenizer.src_lang = lang_code

        encoded = tokenizer(
            line,
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])

        return [
            tok
            for tok in tokens
            if not tok.startswith("__") and tok not in tokenizer.all_special_tokens
        ]

    with open(src_file, "r", encoding="utf-8") as f_src, open(
        tgt_file, "r", encoding="utf-8"
    ) as f_tgt, open(out_file, "w", encoding="utf-8") as f_out:

        for src_line, tgt_line in zip(f_src, f_tgt):
            src_line = src_line.strip()
            tgt_line = tgt_line.strip()

            if not src_line or not tgt_line:
                continue

            src_tokens = tokenize_line(src_line, src_lang)
            tgt_tokens = tokenize_line(tgt_line, tgt_lang)

            f_out.write(" ".join(src_tokens) + " ||| " + " ".join(tgt_tokens) + "\n")


def is_contiguous(idxs):
    idxs = sorted(idxs)
    return idxs == list(range(idxs[0], idxs[-1] + 1))


def build_replacement_plans(
    src_ids,
    tgt_ids,
    alignment,
    random_mask,
):
    """
    Returns a list of replacement plans:
    {
        src_start: int,
        src_len: int,
        tgt_ids: List[int]
    }
    """
    plans = []

    for src_idx, tgt_idxs in alignment.items():
        if src_idx >= len(src_ids):
            continue
        if not random_mask[src_idx]:
            continue

        # 1 → 1
        if len(tgt_idxs) == 1:
            j = tgt_idxs[0]
            if j < len(tgt_ids):
                plans.append(
                    {
                        "src_start": src_idx,
                        "src_len": 1,
                        "tgt_ids": [tgt_ids[j]],
                    }
                )

        # 1 → many (only if contiguous)
        elif is_contiguous(tgt_idxs):
            tgt_idxs = sorted(tgt_idxs)
            plans.append(
                {
                    "src_start": src_idx,
                    "src_len": 1,
                    "tgt_ids": [tgt_ids[j] for j in tgt_idxs if j < len(tgt_ids)],
                }
            )

    return plans


def filter_overlapping_plans(plans):
    plans = sorted(plans, key=lambda p: p["src_start"])
    filtered = []
    occupied = set()

    for p in plans:
        span = set(range(p["src_start"], p["src_start"] + p["src_len"]))
        if occupied & span:
            continue
        filtered.append(p)
        occupied |= span

    return filtered


def apply_replacements(src_ids, plans):
    output = []
    cursor = 0

    for p in plans:
        start = p["src_start"]
        output.extend(src_ids[cursor:start])
        output.extend(p["tgt_ids"])
        cursor = start + p["src_len"]

    output.extend(src_ids[cursor:])
    return output


def collate_and_pad(batch, pad_id):
    max_len = max(len(seq) for seq in batch)
    padded = [seq + [pad_id] * (max_len - len(seq)) for seq in batch]
    attention_mask = [[1] * len(seq) + [0] * (max_len - len(seq)) for seq in batch]
    return {
        "input_ids": torch.tensor(padded),
        "attention_mask": torch.tensor(attention_mask),
    }


def build_fast_align_dict_from_raw(
    src_file: str,
    tgt_file: str,
    tokenizer,
    src_lang: str,
    tgt_lang: str,
    fast_align_bin: str = "/mnt/storage/henok/thesis-henok/fast_align/build/fast_align",
) -> Dict[int, Dict[int, List[int]]]:
    """
    Returns:
        {
          sent_id: {
            src_tok_idx: [tgt_tok_idx, ...]
          }
        }
    """

    def tokenize_line(line: str, lang_code: str) -> List[str]:
        tokenizer.src_lang = lang_code
        encoded = tokenizer([line])  # pass as list to maintain batch dimension
        input_ids = encoded["input_ids"].squeeze(0).tolist()
        toks = tokenizer.convert_ids_to_tokens(input_ids)
        # remove special tokens and language tokens
        return [
            t
            for t in toks
            if not t.startswith("__") and t not in tokenizer.get_special_tokens()
        ]

    # Step 1: create fast_align input
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
        tmp_path = tmp.name
        with open(src_file, encoding="utf-8") as fs, open(
            tgt_file, encoding="utf-8"
        ) as ft:
            i = 0
            for src_line, tgt_line in zip(fs, ft):
                src_line = src_line.strip()
                tgt_line = tgt_line.strip()
                if not src_line or not tgt_line:
                    continue

                src_toks = tokenize_line(src_line, src_lang)
                # print(i)
                # print(src_toks)
                tgt_toks = tokenize_line(tgt_line, tgt_lang)
                # print(tgt_toks)
                i += 1

                tmp.write(" ".join(src_toks) + " ||| " + " ".join(tgt_toks) + "\n")

    # Step 2: run fast_align
    proc = subprocess.run(
        [
            fast_align_bin,
            "-i",
            tmp_path,
            "-d",
            "-o",
            "-v",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=True,
    )

    # Step 3: parse output
    alignments = []
    for line in proc.stdout.strip().splitlines():
        alignment = []
        max_i = -1
        for pair in line.split():
            i, j = pair.split("-")
            i, j = int(i), int(j)
            alignment.append((i, j))
            i = max(max_i, i)
        align_map = [[] for _ in range(i + 1)]
        for i, j in alignment:
            align_map[i].append(j)
        alignments.append(alignment)

    # alignments: Dict[int, Dict[int, List[int]]] = {}
    # for sent_id, line in enumerate(proc.stdout.strip().splitlines()):
    #     sent_align = defaultdict(list)
    #     for pair in line.split():
    #         i, j = pair.split("-")
    #         sent_align[int(i)].append(int(j))
    #     alignments[sent_id] = dict(sent_align)

    return alignments


if __name__ == "__main__":
    from tokenization import NllbTokenizer

    tokenizer = NllbTokenizer("600M")
    map = build_fast_align_dict_from_raw(
        "test_files/lang1.txt",
        "test_files/lang2.txt",
        tokenizer,
        "eng_Latn",
        "fra_Latn",
    )
    for key in map:
        print(f"{key}: {map[key]}")
