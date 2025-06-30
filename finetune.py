import argparse
import gc
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import sys
import torch
from tqdm import tqdm
from transformers import (
    Adafactor,
    AutoModelForSeq2SeqLM,
    get_constant_schedule_with_warmup,
)

from configure import USE_CUDA
from corpora import MixtureOfBitexts, TokenizedMixtureOfBitexts, load_tokenizer
from permutations import (
    create_random_permutation_with_fixed_points,
    save_permutation_map,
)


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()




def prepare_model(base_model: str, freeze_encoder: bool):
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
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
):
    print(f"Training {model_dir}")
    model = prepare_model(base_model, freeze_encoder)
    optimizer = Adafactor(
        [p for p in model.parameters() if p.requires_grad],
        scale_parameter=False,
        relative_step=False,
        lr=1e-4,
        clip_threshold=1.0,
        weight_decay=1e-3,
    )
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=1000)

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
            scheduler.step()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("GPU OOM. Cleaning up.")
                optimizer.zero_grad(set_to_none=True)
                cleanup()
                continue
            else:
                raise

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
        "--model_dir", type=str, required=True, help="Directory to save finetuned model"
    )
    parser.add_argument("--steps", type=int, default=60_000, help="Training steps")
    args = parser.parse_args()

    # Create unique model directory
    base_dir = args.model_dir
    model_version = 0
    while os.path.exists(f"{base_dir}-v{model_version}"):
        model_version += 1
    model_dir = f"{base_dir}-v{model_version}"
    os.makedirs(model_dir)
    train_data = MixtureOfBitexts.create_from_files(
        {
            "fra_Latn": "data/train.fr",
            "deu_Latn": "data/train.de",
            "eng_Latn": "data/train.en",
        },
        [("eng_Latn", "fra_Latn"), ("eng_Latn", "deu_Latn")],
        batch_size=64,
    )
    dev_data = MixtureOfBitexts.create_from_files(
        {
            "fra_Latn": "data/dev.fr",
            "deu_Latn": "data/dev.de",
            "eng_Latn": "data/dev.en",
        },
        [("eng_Latn", "fra_Latn"), ("eng_Latn", "deu_Latn")],
        batch_size=64,
    )
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = load_tokenizer(model_name)
    pmap = {
        "fra_Latn": create_random_permutation_with_fixed_points(
            len(tokenizer), tokenizer.all_special_ids
        ),
        "deu_Latn": create_random_permutation_with_fixed_points(
            len(tokenizer), tokenizer.all_special_ids
        )
    }
    save_permutation_map(pmap, Path(model_dir) / "permutations.json")
    tokenized_train = TokenizedMixtureOfBitexts(
        train_data, tokenizer, max_length=128, permutation_map=pmap
    )
    tokenized_dev = TokenizedMixtureOfBitexts(
        dev_data, tokenizer, max_length=128, permutation_map=pmap
    )
    finetune(
        tokenized_train,
        tokenized_dev,
        model_name,
        model_dir,
        args.steps,
        freeze_encoder=False,
    )

    test_mix = MixtureOfBitexts.create_from_files(
        {
            "fra_Latn": "data/test.fr",
            "deu_Latn": "data/test.de",
            "eng_Latn": "data/test.en",
        },
        [("eng_Latn", "fra_Latn"), ("eng_Latn", "deu_Latn")],
        batch_size=32,
        only_once_thru=True,
    )
    tokenized_test = TokenizedMixtureOfBitexts(test_mix, tokenizer, max_length=128)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    if USE_CUDA:
        model.cuda()
    #translations = translate_tokenized_mixture_of_bitexts(
    #    tokenized_test, model, tokenizer, pmap
    #)
    #with open(Path(model_dir) / "translations.json", "w") as writer:
    #    json.dump(translations, writer)
    #print("Translations complete.")

    test_mix = MixtureOfBitexts.create_from_files(
        {
            "fra_Latn": "data/test.fr",
            "deu_Latn": "data/test.de",
            "eng_Latn": "data/test.en",
        },
        [("eng_Latn", "fra_Latn"), ("eng_Latn", "deu_Latn")],
        batch_size=32,
        only_once_thru=True,
    )
    references = dict()
    batch = test_mix.next_batch()
    while batch is not None:
        _, tgt, src_code, tgt_code = batch
        key = "->".join([src_code, tgt_code])
        if key not in references:
            references[key] = []
        references[key].extend(tgt)
        batch = test_mix.next_batch()
    with open(Path(model_dir) / "references.json", "w") as writer:
        json.dump(references, writer)
    print("References complete.")


    #scores = dict()
    #for key in translations:
    #    scores[key] = evaluate_translations(
    #        translations[key], 
    #        references[key]
    #    )
    #with open(Path(model_dir) / "scores.json", "w") as writer:
    #    json.dump(scores, writer)
    #print("Evaluation complete.")    


if __name__ == "__main__":
    main()
