USE_CUDA = True

from dataclasses import dataclass
import json
import os
from pathlib import Path
from permutations import (
    create_random_permutation_with_fixed_points,
    load_permutation_map,
)
import shutil
from tokenization import NllbTokenizer, HuggingfaceTokenizer, ByteTokenizer
from transformers import AutoTokenizer
from corpora import (
    Corpus,
    TokenizedCorpus,
    EncipheredCorpus,
    Bitext,
    MixtureOfBitexts,
    CodeswitchedBitext,
    BatchedBitext,
)
from compressor import load_autocompleting_tokenizer


@dataclass
class FinetuningParameters:
    base_model: str
    should_finetune: bool
    report_every: int
    validate_every: int
    patience: int
    batch_size: int
    num_training_steps: int
    freeze_encoder: bool
    freeze_decoder: bool
    gradient_accumulation_steps: int
    max_grad_norm: float
    dev_batches: int


def read_finetuning_params(config):
    """Reads the finetuning parameters into a dataclass."""
    params = config["finetuning_parameters"]
    f_params = FinetuningParameters(
        base_model=params["base_model"],
        should_finetune=params.get("finetune", True),
        report_every=params.get("report_every", 500),
        validate_every=params.get("validate_every", 500),
        patience=params.get("patience", 1000000000),
        batch_size=params["batch_size"],
        num_training_steps=params["num_steps"],
        freeze_decoder=params.get("freeze_decoder", False),
        freeze_encoder=params.get("freeze_encoder", False),
        gradient_accumulation_steps=params.get("gradient_accumulation_steps", 1),
        max_grad_norm=params.get("max_grad_norm", 1.0),
        dev_batches=params.get("dev_batches", 100),
    )
    return f_params


def create_experiment_dir(config, config_file):
    """Creates a new experiment directory and copies the config file into it."""
    base_dir = config["model_dir"]
    model_version = 0
    while os.path.exists(f"{base_dir}-v{model_version}"):
        model_version += 1
    model_dir = f"{base_dir}-v{model_version}"
    os.makedirs(model_dir)
    shutil.copy(config_file, Path(model_dir) / "experiment.json")
    return model_dir


def harvest_language_codes(config):
    """Creates a dictionary that maps (corpus, lang) pairs to language codes."""
    lang_codes = dict()
    for corpus in config["corpora"]:
        for key in config["corpora"][corpus]:
            lang_codes[(corpus, key)] = config["corpora"][corpus][key]["lang_code"]
    return lang_codes


# TODO: update
def initialize_tokenizer(config):
    # TODO: generalize to separate src/tgt tokenizers
    params = config["finetuning_parameters"]
    model_name = params["base_model"]
    if model_name == "facebook/nllb-200-distilled-600M":
        tokenizer = NllbTokenizer("600M", max_length=128)  # set max length?
    elif model_name == "facebook/nllb-200-distilled-1.3B":
        tokenizer = NllbTokenizer("1.3B", max_length=128)
    else:
        tokenizer = HuggingfaceTokenizer(model_name, max_length=128)
    return tokenizer


def create_ciphers(config, tokenizer_map):
    all_corpora = config["corpora"]
    ciphers = dict()
    cipher_map = dict()
    for corpus_name in all_corpora:
        cipher_index = all_corpora[corpus_name]["encipherment"]
        tokenizer_name = all_corpora[corpus_name]["tokenizer"]
        tokenizer = tokenizer_map[tokenizer_name]
        if cipher_index != "0":
            cipher_id = (tokenizer_name, cipher_index)
            if cipher_id not in ciphers:
                ciphers[cipher_id] = create_random_permutation_with_fixed_points(
                    len(tokenizer),
                    list(tokenizer.get_special_tokens().values()),
                )
            cipher_map[(tokenizer_name, cipher_index)] = ciphers[cipher_id]
    return cipher_map


def create_bitexts(config, cipher_map=None):
    tokenizer_map = dict()
    for tokenizer_name in config["tokenizers"]:
        tokenizer_config = config["tokenizers"][tokenizer_name]
        if tokenizer_config["type"] == "huggingface":
            tokenizer = HuggingfaceTokenizer(
                tokenizer_config["model"], max_length=tokenizer_config["max_length"]
            )
        elif tokenizer_config["type"] == "byte":
            tokenizer = ByteTokenizer(max_length=tokenizer_config["max_length"])
        elif tokenizer_config["type"] == "autocompleting":
            tokenizer = load_autocompleting_tokenizer(
                tokenizer_config["model"],
                max_length=tokenizer_config["max_length"],
                max_length_encoding=(
                    tokenizer_config["max_length_encoding"]
                    if "max_length_encoding" in tokenizer_config
                    else 1
                ),
            )
        else:
            raise Exception(f"Unrecognized tokenizer type: {tokenizer_config["type"]}")
        tokenizer_map[tokenizer_name] = tokenizer
    if cipher_map is None:
        cipher_map = create_ciphers(config, tokenizer_map)
    all_corpora = dict()
    for corpus_name in config["corpora"]:
        corpus_config = config["corpora"][corpus_name]
        for split in ["train", "dev", "test"]:
            tokenizer = tokenizer_map[corpus_config["tokenizer"]]
            text_file = corpus_config[split]
            lang_code = corpus_config["lang_code"]
            tokenizer_name = corpus_config["tokenizer"]
            encipherment = corpus_config["encipherment"]
            corpus = TokenizedCorpus(Corpus(text_file), tokenizer, lang_code)
            if (tokenizer_name, encipherment) in cipher_map:
                corpus = EncipheredCorpus(
                    corpus, cipher_map[(tokenizer_name, encipherment)]
                )
            all_corpora[(corpus_name, split)] = corpus

    bitexts = dict()
    metadata = dict()
    params = config["finetuning_parameters"]
    for bitext in config["bitexts"]:
        src = bitext["src"]
        tgt = bitext["tgt"]
        src_config = config["corpora"][src]
        tgt_config = config["corpora"][tgt]
        bitexts[(src, tgt)] = dict()
        metadata[(src, tgt)] = {
            "lang1_tokenizer": src_config["tokenizer"],
            "lang1_encipherment": src_config["encipherment"],
            "lang1_code": src_config["lang_code"],
            "lang2_tokenizer": tgt_config["tokenizer"],
            "lang2_encipherment": tgt_config["encipherment"],
            "lang2_code": tgt_config["lang_code"],
        }
        for split in ["train", "dev", "test"]:
            lines = (
                bitext["train_lines"]
                if split == "train" and "train_lines" in bitext
                else None
            )
            bitexts[(src, tgt)][split] = BatchedBitext(
                Bitext(all_corpora[(src, split)], all_corpora[(tgt, split)], lines),
                params["batch_size"],
                params["src_pad_id"],
                params["tgt_pad_id"],
            )

    mixtures = dict()
    for split in ["train", "dev", "test"]:
        split_bitexts = {key: bitexts[key][split] for key in bitexts}
        mixtures[split] = MixtureOfBitexts(
            split_bitexts,
            metadata,
            sampling_probs=None,
            only_once_thru=(split != "train"),
        )
    mixtures["cipher_map"] = cipher_map
    mixtures["tokenizer_map"] = tokenizer_map
    return mixtures


def create_bitexts_from_experiment_dir(experiment_dir):
    config_file = Path(experiment_dir) / "experiment.json"
    with open(config_file) as reader:
        config = json.load(reader)
    emap = load_permutation_map(Path(experiment_dir) / "ciphers.json")
    return create_bitexts(config, cipher_map=emap)
