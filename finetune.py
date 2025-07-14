import argparse
import gc
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import shutil
import sys
import torch
from tqdm import tqdm
from transformers import (
    Adafactor,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    get_constant_schedule_with_warmup,
)

from configure import USE_CUDA
from corpora import MixtureOfBitexts, TokenizedMixtureOfBitexts, load_tokenizer
from permutations import (
    create_random_permutation_with_fixed_points,
    save_permutation_map,
)
from validate import translate_tokenized_mixture_of_bitexts, evaluate_translations


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def prepare_model(base_model: str, freeze_encoder: bool, should_finetune: bool):
    if should_finetune:
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model) 
        print('loaded pretrained model')
    else: 
        model_config = AutoConfig.from_pretrained(base_model)
        model = AutoModelForSeq2SeqLM.from_config(model_config)
        print('loaded architecture only')
    if hasattr(model.config, "max_length"):  # this should be in a GenerationConfig
        delattr(model.config, "max_length")
    if freeze_encoder:
        print("--> ENCODER FROZEN <--")
        for param in model.get_encoder().parameters():
            param.requires_grad = False
    else:
        print("--> encoder NOT frozen <--")
    if USE_CUDA:
        torch.cuda.set_device(0)
        model.cuda()
    return model


def evaluate(model, dev_data, batches: int = 100):
    model.eval()
    dev_losses = []
    with torch.no_grad():
        for _ in range(batches):
            x, y, _, _ = dev_data.next_batch()
            x = x.to(model.device)
            y = y.to(model.device)
            loss = model(**x, labels=y.input_ids).loss
            dev_losses.append(loss.item())
    return np.mean(dev_losses)


def plot_losses(train_x, train_y, dev_x, dev_y, out_path: str):
    plt.clf()
    plt.plot(train_x, train_y, label="train", color="blue", linewidth=2)
    plt.plot(dev_x, dev_y, label="dev", color="red", linewidth=2)
    plt.xlabel("training steps")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path)


def finetune(
    train_data,
    dev_data,
    base_model: str,
    model_dir: str,
    training_steps: int,
    report_every: int = 500,
    validate_every: int = 500,
    patience: int = 5,
    freeze_encoder: bool = False,
    should_finetune: bool = True
):
    print(f"Training {model_dir}")
    model = prepare_model(base_model, freeze_encoder, should_finetune)
    
    if should_finetune:
        optimizer = Adafactor(
            [p for p in model.parameters() if p.requires_grad],
            scale_parameter=False,
            relative_step=False,
            lr=1e-4,
            clip_threshold=1.0,
            weight_decay=1e-3,
        )
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=1000)
    else:
        optimizer = Adafactor(
            model.parameters(),
            scale_parameter=True,
            relative_step=True,
            lr=None,  # Required when using relative_step
            clip_threshold=1.0,
            weight_decay=0.01,  
        )
        scheduler = None
        
    cleanup()
    train_losses, train_plot_x, train_plot_y = [], [], []
    dev_plot_x, dev_plot_y = [], []
    best_dev_loss, steps_since_best = None, 0

    for i in tqdm(range(training_steps)):
        try:
            model.train()
            x, y, _, _ = train_data.next_batch()
            x = x.to(model.device)
            y = y.to(model.device)
            loss = model(**x, labels=y.input_ids).loss
            loss.backward()
            train_losses.append(loss.item())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("GPU OOM. Cleaning up.")
                optimizer.zero_grad(set_to_none=True)
                cleanup()
                continue
            else:
                raise e

        if i > 0 and i % report_every == 0:
            avg_train_loss = np.mean(train_losses[-report_every:])
            print(f"Step {i} (train): {avg_train_loss:.4f}")
            train_plot_x.append(i)
            train_plot_y.append(avg_train_loss)
            sys.stdout.flush()

        if i % validate_every == 0:
            print("Validating...")
            dev_loss = evaluate(model, dev_data)
            print(f"Dev loss: {dev_loss:.4f}")
            dev_plot_x.append(i)
            dev_plot_y.append(dev_loss)
            sys.stdout.flush()

            plot_losses(
                train_plot_x,
                train_plot_y,
                dev_plot_x,
                dev_plot_y,
                os.path.join(model_dir, "training.png"),
            )

            if best_dev_loss is None or dev_loss < best_dev_loss:
                print("Saving new best model.")
                best_dev_loss = dev_loss
                steps_since_best = 0
                model.save_pretrained(model_dir)  # causes warning?
            else:
                steps_since_best += 1
                print(f"No improvement. Patience: {patience - steps_since_best}")
                if steps_since_best >= patience:
                    print("Early stopping.")
                    break


def main():
    parser = argparse.ArgumentParser(description="Finetune NLLB model.")
    parser.add_argument(
        "--config", type=str, required=True, help="Directory to save finetuned model"
    )
    args = parser.parse_args()

    with open(args.config) as reader:
        config = json.load(reader)

    all_corpora = config["corpora"]
    params = config["finetuning_parameters"]
    should_finetune = params["finetune"] if "finetune" in params else True
    
    # Create unique model directory
    base_dir = config["model_dir"]
    model_version = 0
    while os.path.exists(f"{base_dir}-v{model_version}"):
        model_version += 1
    model_dir = f"{base_dir}-v{model_version}"
    os.makedirs(model_dir)
    shutil.copy(args.config, Path(model_dir) / Path(args.config).name)

    lang_codes = dict()        
    for corpus in config['corpora']:
        for key in config['corpora'][corpus]:
            print(key)
            lang_codes[(corpus, key)] = config['corpora'][corpus][key]['lang_code']
    

    train_data = MixtureOfBitexts.create_from_config(config, "train", only_once_thru=False)    
    dev_data = MixtureOfBitexts.create_from_config(config, "dev", only_once_thru=False)
    model_name = params["base_model"]
    tokenizer = load_tokenizer(model_name)

    # Create the permutations
    permutations = dict()
    pmap = dict()
    for corpus in all_corpora:
        for language in all_corpora[corpus]:
            permutation_index = all_corpora[corpus][language]["permutation"]
            if permutation_index > 0:
                if permutation_index not in permutations:
                    permutations[permutation_index] = (
                        create_random_permutation_with_fixed_points(
                            len(tokenizer), tokenizer.all_special_ids
                        )
                    )
                pmap[(corpus, language)] = permutations[permutation_index]
        
    save_permutation_map(pmap, Path(model_dir) / "permutations.json")
    tokenized_train = TokenizedMixtureOfBitexts(
        train_data, tokenizer, max_length=128, lang_codes=lang_codes, permutation_map=pmap
    )
    tokenized_dev = TokenizedMixtureOfBitexts(
        dev_data, tokenizer, max_length=128, lang_codes=lang_codes, permutation_map=pmap
    )
    finetune(
        tokenized_train,
        tokenized_dev,
        model_name,
        model_dir,
        params['num_steps'],
        freeze_encoder=params['freeze_encoder'] if 'freeze_encoder' in params else False,
        should_finetune=should_finetune
    )

    test_data = MixtureOfBitexts.create_from_config(config, "test", only_once_thru=True)    
    tokenized_test = TokenizedMixtureOfBitexts(test_data, tokenizer, max_length=128, lang_codes=lang_codes, permutation_map=pmap)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    if USE_CUDA:
        model.cuda()
    translations = translate_tokenized_mixture_of_bitexts(
        tokenized_test, model, tokenizer, lang_codes, pmap
    )
    with open(Path(model_dir) / "translations.json", "w") as writer:
        json.dump(translations, writer)
    print("Translations complete.")

    test_data = MixtureOfBitexts.create_from_config(config, "test", only_once_thru=True)    
    references = dict()
    batch = test_data.next_batch()
    while batch is not None:
        _, tgt, src_lang, tgt_lang = batch
        src_code = lang_codes[src_lang]
        tgt_code = lang_codes[tgt_lang]
        key = '->'.join([src_code, tgt_code])
        if key not in references:
            references[key] = []
        references[key].extend(tgt)
        batch = test_data.next_batch()
    with open(Path(model_dir) / "references.json", "w") as writer:
        json.dump(references, writer)
    print("References complete.")

    scores = dict()
    for key in translations:
        scores[key] = evaluate_translations(translations[key], references[key])
    with open(Path(model_dir) / "scores.json", "w") as writer:
        json.dump(scores, writer)
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
