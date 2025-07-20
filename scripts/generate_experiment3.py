import json
from pathlib import Path

VARIANT = 5

if VARIANT == 1:
    BASE_MODEL = "facebook/nllb-200-distilled-600M"
    SRC = "en"
    SRC_ID = "eng_Latn"
    TGTS = ["es", "pt"]
    FREEZE_ENCODER = True
elif VARIANT == 2:
    BASE_MODEL = "facebook/nllb-200-distilled-600M"
    SRC = "en"
    SRC_ID = "eng_Latn"
    TGTS = ["es", "pt"]
    FREEZE_ENCODER = True
elif VARIANT == 3:
    BASE_MODEL = "facebook/nllb-200-distilled-600M"
    SRC = "en"
    SRC_ID = "eng_Latn"
    TGTS = ["es", "pt"]
    FREEZE_ENCODER = True
elif VARIANT == 4:
    BASE_MODEL = "facebook/nllb-200-distilled-600M"
    SRC = "en"
    SRC_ID = "eng_Latn"
    TGTS = ["es", "pt"]
    FREEZE_ENCODER = True
elif VARIANT == 5:
    BASE_MODEL = "facebook/nllb-200-distilled-600M"
    SRC = "en"
    SRC_ID = "eng_Latn"
    TGTS = ["es", "pt"]
    FREEZE_ENCODER = True


TGT_IDS = [
    "tsn_Latn",
    "tso_Latn",
    "tuk_Latn",
    "tum_Latn",
    "tur_Latn",
    "twi_Latn",
    "umb_Latn",
    "uzn_Latn",
    "vec_Latn",
    "vie_Latn",
    "war_Latn",
    "wol_Latn",
    "xho_Latn",
]


def create_bituning_config(num_train_lines, tgt_index):
    return {
        "model_dir": f"experiments/exp3-{VARIANT}/exp3-{VARIANT}-bi{tgt_index}-{num_train_lines}",
        "corpora": { 
            "europarl": {
                SRC: {
                    "lang_code": SRC_ID,
                    "train": f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/optimized_data/optimized_train_128.{SRC}",
                    "dev": f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/europarlData/dev.{SRC}",
                    "test": f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/europarlData/test.{SRC}",
                    "permutation": 0,
                },
                TGTS[tgt_index]: {
                    "lang_code": TGT_IDS[tgt_index],
                    "train": f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/optimized_data/optimized_train_128.{TGTS[tgt_index]}",
                    "dev": f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/europarlData/dev.{TGTS[tgt_index]}",
                    "test": f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/europarlData/test.{TGTS[tgt_index]}",
                    "permutation": 1,
                },
            }
        },
        "bitexts": [
            {
                "corpus": "europarl",
                "src": SRC,
                "tgt": TGTS[tgt_index],
                "train_lines": [
                    tgt_index * num_train_lines,
                    tgt_index * num_train_lines + num_train_lines,
                ],
            }
        ],
        "finetuning_parameters": {
            "base_model": BASE_MODEL,
            "batch_size": 64,
            "num_steps": 60000,
            "freeze_encoder": FREEZE_ENCODER
        },
    }


def create_multituning_config(num_train_lines):
    corpora = { 
            "europarl": {
                SRC: {
                    "lang_code": SRC_ID,
                    "train": f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/optimized_data/optimized_train_128.{SRC}",
                    "dev": f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/europarlData/dev.{SRC}",
                    "test": f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/europarlData/test.{SRC}",
                    "permutation": 0,
                }
            },
            "lexicon": {
                TGTS[0]: {
                    "lang_code": TGT_IDS[0],
                    "train": f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/scripts/common_vocab/train_tokenized/train.{TGTS[0]}_{TGTS[1]}_common_vocab",
                    "dev": f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/scripts/common_vocab/dev_tokenized/dev.{TGTS[0]}_{TGTS[1]}_common_vocab",
                    "test": f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/scripts/common_vocab/test_tokenized/test.{TGTS[0]}_{TGTS[1]}_common_vocab",
                    "permutation": 1,
                }
            }
        }
    
    corpora["europarl"][TGTS[0]] = {
        "lang_code": TGT_IDS[0],
        "train": f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/optimized_data/optimized_train_128.{TGTS[0]}",
        "dev": f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/europarlData/dev.{TGTS[0]}",
        "test": f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/europarlData/test.{TGTS[0]}",
        "permutation": 1,
    }
    corpora["lexicon"][TGTS[1]] = {
        "lang_code": TGT_IDS[1],
        "train": f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/scripts/common_vocab/train_tokenized/train.{TGTS[0]}_{TGTS[1]}_common_vocab",
        "dev": f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/scripts/common_vocab/dev_tokenized/dev.{TGTS[0]}_{TGTS[1]}_common_vocab",
        "test": f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/scripts/common_vocab/test_tokenized/test.{TGTS[0]}_{TGTS[1]}_common_vocab",
        "permutation": 1,
    }
    bitexts = [
        {
            "corpus": "europarl",
            "src": SRC,
            "tgt": TGTS[0],
            "train_lines": [
                0 * num_train_lines,
                0 * num_train_lines + num_train_lines,
            ],
        },
        {
            "corpus": "lexicon",
            "src": TGTS[0],
            "tgt": TGTS[1],
            "train_lines": [
                1 * num_train_lines,
                1 * num_train_lines + num_train_lines,
            ],
        }
    ]
    return {
        "model_dir": f"experiments/exp3-{VARIANT}/exp3-{VARIANT}-multi-{num_train_lines}",
        "corpora": corpora,
        "bitexts": bitexts,
        "finetuning_parameters": {
            "base_model": BASE_MODEL,
            "batch_size": 64,
            "num_steps": 60000,
            "freeze_encoder": FREEZE_ENCODER
        },
    }


def create_shell_script(num_train_lines):
    preface = [
            "#!/bin/sh",
            "#SBATCH -c 1",
            "#SBATCH -t 3-12:00",
            "#SBATCH -p dl",
            "#SBATCH -o logs/log_%j.out",
            "#SBATCH -e logs/log_%j.err",
            "#SBATCH --gres=gpu:1",
        ]
    exp_config = config_dir / f"experiment3-{VARIANT}.multi.{num_train_lines}.json"
    preface.append(f"python finetune_dev.py --config {exp_config}")
    for tgt_index in range(1):
        exp_config = config_dir / f"experiment3-{VARIANT}.bi{tgt_index}.{num_train_lines}.json"
        preface.append(f"python finetune_dev.py --config {exp_config}",)
    return "\n".join(preface)

config_dir = Path(f"configs/exp3-{VARIANT}")
config_dir.mkdir(parents=True, exist_ok=True)
for num_train_lines in [1024, 2048, 4096, 8192, 16834]:
    for tgt_index in range(1):
        config = create_bituning_config(num_train_lines, tgt_index)
        with open(
            config_dir / f"experiment3-{VARIANT}.bi{tgt_index}.{num_train_lines}.json", "w"
        ) as writer:
            json.dump(config, writer, indent=4)
    config = create_multituning_config(num_train_lines)
    with open(
        config_dir / f"experiment3-{VARIANT}.multi.{num_train_lines}.json", "w"
    ) as writer:
        json.dump(config, writer, indent=4)
    shell_script = create_shell_script(num_train_lines)
    with open(config_dir / f"run.exp3-{VARIANT}.{num_train_lines}.sh", "w") as writer:
        writer.write(shell_script)
