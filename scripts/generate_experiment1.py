import json
from pathlib import Path

VARIANT = 4

# defaults
BASE_MODEL = "facebook/nllb-200-distilled-600M"
DATA_DIR = "/mnt/storage/sotnichenko/encoder-decoder-finetuning/ENG_ZH"
FREEZE_ENCODER = True
FREEZE_DECODER = False
SRC = "en"
SRC_ID = "eng_Latn"

if VARIANT == 1:
    FREEZE_ENCODER = False
    TGT = "de"
    TGTS = [f'{TGT}{i}' for i in range(2)]
elif VARIANT == 4:
    TGT = "zh"
    TGTS = [f'{TGT}{i}' for i in range(2)]

elif VARIANT == 5:
    TGT = "sr"
    TGTS = [f'{TGT}{i}' for i in range(2)]



FT_PARAMS = {
    "base_model": BASE_MODEL,
    "freeze_encoder": FREEZE_ENCODER,
    "freeze_decoder": FREEZE_DECODER,
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
            "ENG_ZH": {
                SRC: {
                    "lang_code": SRC_ID,
                    "train": f"{DATA_DIR}/train.{TGT}",
                    "dev": f"{DATA_DIR}/dev.{TGT}",
                    "test": f"{DATA_DIR}/test.{TGT}",
                    "permutation": 0,
                },
                TGTS[tgt_index]: {
                    "lang_code": TGT_IDS[tgt_index],
                    "train": f"{DATA_DIR}/train.{TGT}",
                    "dev": f"{DATA_DIR}/dev.{TGT}",
                    "test": f"{DATA_DIR}/test.{TGT}",
                    "permutation": 1,
                }
            }
        },
        "bitexts": [
            {
                "corpus": "ENG_ZH",
                "src": SRC,
                "tgt": TGTS[tgt_index],
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
        "ENG_ZH": {
            SRC: {
                "lang_code": SRC_ID,
                "train": f"{DATA_DIR}/train.{TGT}",
                "dev": f"{DATA_DIR}/dev.{TGT}",
                "test": f"{DATA_DIR}/test.{TGT}",
                "permutation": 0,
            }
        },
        "ENG_ZH": {
            SRC: {
                "lang_code": SRC_ID,
                "train": f"{DATA_DIR}/train.{TGT}",
                "dev": f"{DATA_DIR}/dev.{TGT}",
                "test": f"{DATA_DIR}/test.{TGT}",
                "permutation": 0,
            }

        }
   
    }
    for tgt_index, tgt in enumerate(TGTS):
        corpora[f"ENG_ZH"][TGTS[tgt_index]] = {
            "lang_code": TGT_IDS[tgt_index],
            "train": f"{DATA_DIR}/train.{TGT}",
            "dev": f"{DATA_DIR}/dev.{TGT}",
            "test": f"{DATA_DIR}/test.{TGT}",
            "permutation": 1 + tgt_index,
        }
    bitexts = [
        {
            "corpus": f"ENG_ZH",
            "src": SRC,
            "tgt": TGTS[tgt_index],
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
for num_train_lines in [1024, 2048, 4096, 8192, 16384]:
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