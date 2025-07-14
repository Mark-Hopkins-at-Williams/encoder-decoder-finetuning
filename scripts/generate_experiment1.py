import json
from pathlib import Path

VARIANT = 4


if VARIANT == 1:
    BASE_MODEL = "facebook/nllb-200-distilled-600M"
    FREEZE_ENCODER = False
    SRC = "en"
    SRC_ID = "eng_Latn"
    TGTS = ["de"] * 2
elif VARIANT == 2:
    BASE_MODEL = "facebook/nllb-200-distilled-600M"
    FREEZE_ENCODER = False
    SRC = "en"
    SRC_ID = "eng_Latn"
    TGTS = ["es"] * 2
elif VARIANT == 3:
    BASE_MODEL = "experiments/eng-jpn-pretrained-v0/"
    FREEZE_ENCODER = False
    SRC = "en"
    SRC_ID = "eng_Latn"
    TGTS = ["es"] * 2
elif VARIANT == 4:
    BASE_MODEL = "experiments/eng-jpn-pretrained-v0/"
    FREEZE_ENCODER = True
    SRC = "en"
    SRC_ID = "eng_Latn"
    TGTS = ["es"] * 2

FT_PARAMS = {
    "base_model": BASE_MODEL,
    "freeze_encoder": FREEZE_ENCODER,
    "batch_size": 64,
    "num_steps": 60000,
}

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
        "model_dir": f"experiments/exp1-{VARIANT}/exp1-{VARIANT}-bi{tgt_index}-{num_train_lines}",
        "corpora": {
            SRC_ID: {
                "train": f"data/train.{SRC}",
                "dev": f"data/dev.{SRC}",
                "test": f"data/test.{SRC}",
                "permutation": 0,
            },
            TGT_IDS[tgt_index]: {
                "train": f"data/train.{TGTS[tgt_index]}",
                "dev": f"data/dev.{TGTS[tgt_index]}",
                "test": f"data/test.{TGTS[tgt_index]}",
                "permutation": 1,
            },
        },
        "bitexts": [
            {
                "src": SRC_ID,
                "tgt": TGT_IDS[tgt_index],
                "train_lines": [
                    tgt_index * num_train_lines,
                    tgt_index * num_train_lines + num_train_lines,
                ],
            }
        ],
        "finetuning_parameters": FT_PARAMS,
    }


def create_multituning_config(num_train_lines):
    corpora = {
        SRC_ID: {
            "train": f"data/train.{SRC}",
            "dev": f"data/dev.{SRC}",
            "test": f"data/test.{SRC}",
            "permutation": 0,
        }
    }
    for tgt_index, tgt in enumerate(TGTS):
        corpora[TGT_IDS[tgt_index]] = {
            "train": f"data/train.{tgt}",
            "dev": f"data/dev.{tgt}",
            "test": f"data/test.{tgt}",
            "permutation": 1 + tgt_index,
        }
    bitexts = [
        {
            "src": SRC_ID,
            "tgt": TGT_IDS[tgt_index],
            "train_lines": [
                tgt_index * num_train_lines,
                tgt_index * num_train_lines + num_train_lines,
            ],
        }
        for tgt_index in range(len(TGTS))
    ]
    return {
        "model_dir": f"experiments/exp1-{VARIANT}/exp1-{VARIANT}-multi-{num_train_lines}",
        "corpora": corpora,
        "bitexts": bitexts,
        "finetuning_parameters": FT_PARAMS,
    }


def create_shell_script(num_train_lines):
    preface = [
        "#!/bin/sh",
        "#SBATCH -c 1",
        "#SBATCH -t 3-12:00",
        "#SBATCH -p dl",
        # "#SBATCH --mem=10G",
        "#SBATCH -o logs/log_%j.out",
        "#SBATCH -e logs/log_%j.err",
        "#SBATCH --gres=gpu:1",
    ]
    exp_config = config_dir / f"experiment1-{VARIANT}.multi.{num_train_lines}.json"
    preface.append(f"python finetune.py --config {exp_config}")
    for tgt_index in range(len(TGTS)):
        exp_config = (
            config_dir / f"experiment1-{VARIANT}.bi{tgt_index}.{num_train_lines}.json"
        )
        preface.append(
            f"python finetune.py --config {exp_config}",
        )
    return "\n".join(preface)


config_dir = Path(f"configs/exp1-{VARIANT}")
config_dir.mkdir(parents=True, exist_ok=True)
for num_train_lines in [1024, 2048, 4096, 8192, 16834]:
    for tgt_index in range(len(TGTS)):
        config = create_bituning_config(num_train_lines, tgt_index)
        with open(
            config_dir / f"experiment1-{VARIANT}.bi{tgt_index}.{num_train_lines}.json",
            "w",
        ) as writer:
            json.dump(config, writer, indent=4)
    config = create_multituning_config(num_train_lines)
    with open(
        config_dir / f"experiment1-{VARIANT}.multi.{num_train_lines}.json", "w"
    ) as writer:
        json.dump(config, writer, indent=4)
    shell_script = create_shell_script(num_train_lines)
    with open(config_dir / f"run.exp1-{VARIANT}.{num_train_lines}.sh", "w") as writer:
        writer.write(shell_script)
