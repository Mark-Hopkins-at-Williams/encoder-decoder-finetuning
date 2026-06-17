import argparse
from attention import SimpleAttention
from configure import create_experiment_dir
from configure import read_finetuning_params
from configure import create_bitexts
from corpora import MixtureOfTextAndGoalEncodings
from pathlib import Path
from permutations import save_permutation_map
import json
import matplotlib
import matplotlib.pyplot as plt
from myutil import cleanup
from myutil import logger
from myutil import prepare_model_for_finetuning
import numpy as np
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import Adafactor
from transformers import get_constant_schedule_with_warmup
from validate import evaluate_experiment

matplotlib.use("Agg")


def plot_losses(train_x, train_y, dev_x, dev_y, out_path: str):
    plt.clf()
    plt.plot(train_x, train_y, label="train", color="blue", linewidth=2)
    plt.plot(dev_x, dev_y, label="dev", color="red", linewidth=2)
    plt.xlabel("training steps")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path)


def finetune(model, train_data1, dev_data, model_dir, ft_params):
    def compute_loss(batch):
        encoder.eval()
        sents, lang, goal_encodings, goal_attn_mask = batch
        sent_attn_mask = sents["attention_mask"].to(encoder.device)
        sents = {k: v.to(encoder.device) for k, v in sents.items()}
        goal_encodings = goal_encodings.to(encoder.device)
        goal_attn_mask = goal_attn_mask.to(encoder.device)
        sent_encodings = encoder(**sents).last_hidden_state
        if lang != "spa_Latn":
            out1, _ = attn(
                sent_encodings, goal_encodings, sent_attn_mask, goal_attn_mask
            )
            token_scores1 = 1 - F.cosine_similarity(sent_encodings, out1, dim=-1)
            token_scores1 = token_scores1 * sent_attn_mask
            loss1 = token_scores1.sum() / sent_attn_mask.sum()
            out2, _ = attn(
                goal_encodings,
                sent_encodings,
                goal_attn_mask,
                sents["attention_mask"],
            )
            token_scores2 = 1 - F.cosine_similarity(goal_encodings, out2, dim=-1)
            token_scores2 = token_scores2 * goal_attn_mask
            loss2 = token_scores2.sum() / goal_attn_mask.sum()

            token_norm_diffs = (
                torch.norm(out2, dim=-1) - torch.norm(goal_encodings, dim=-1)
            ) ** 2
            loss4 = torch.sqrt(
                (
                    torch.mean(torch.norm(sent_encodings, dim=-1))
                    - torch.mean(torch.norm(goal_encodings, dim=-1))
                )
                ** 2
            )
            loss = loss4 + ((loss1 + loss2) / 2.0)
        else:
            token_scores = 1 - F.cosine_similarity(
                sent_encodings, goal_encodings, dim=-1
            )
            token_scores = token_scores * sent_attn_mask
            loss1 = token_scores.sum() / sent_attn_mask.sum()
            token_norm_diffs = (
                torch.norm(sent_encodings, dim=-1) - torch.norm(goal_encodings, dim=-1)
            ) ** 2
            token_norm_diffs = token_norm_diffs * sent_attn_mask
            loss2 = token_norm_diffs.sum() / sent_attn_mask.sum()
            loss = loss1 + loss2
        return loss

    logger(f"Training {model_dir}")
    model.save_pretrained(model_dir)
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
    encoder = model.model.encoder
    attn = SimpleAttention()
    train_data_iter = iter(train_data1)
    for i in tqdm(range(ft_params.num_training_steps)):
        try:
            loss = compute_loss(next(train_data_iter))
            loss.backward()
            train_losses.append(loss.item())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger("GPU OOM. Cleaning up.", to_stderr=True)
                optimizer.zero_grad(set_to_none=True)
                cleanup()
                continue
            else:
                raise e
        if i > 0 and i % ft_params.report_every == 0:
            avg_train_loss = np.mean(train_losses[-ft_params.report_every :])
            logger(f"Step {i} (train): {avg_train_loss:.4f}")
            train_plot_x.append(i)
            train_plot_y.append(avg_train_loss)
        if i > 0 and i % ft_params.validate_every == 0:
            logger("Validating...")

            def evaluate(dev_data):
                dev_data.restart()
                dev_losses = dict()
                with torch.no_grad():
                    for batch in dev_data:
                        sents, lang, goal_encodings, goal_attn_mask = batch
                        loss = compute_loss(batch)
                        if lang not in dev_losses:
                            dev_losses[lang] = []
                        dev_losses[lang].append(loss.item())

                return {k: np.mean(dev_losses[k]) for k in dev_losses}

            dev_loss = evaluate(dev_data)
            for lang in dev_loss:
                logger(f"Dev loss ({lang}): {dev_loss[lang]:.2f}")
            dev_plot_x.append(i)
            dev_plot_y.append(dev_loss["tsn_Latn"])
            plot_losses(
                train_plot_x,
                train_plot_y,
                dev_plot_x,
                dev_plot_y,
                os.path.join(model_dir, "training.png"),
            )
            if best_dev_loss is None or dev_loss["tsn_Latn"] < best_dev_loss:
                logger("Saving new best model.")
                best_dev_loss = dev_loss["tsn_Latn"]
                steps_since_best = 0
                model.save_pretrained(model_dir)
            else:
                steps_since_best += 1
                logger(
                    f"No improvement. Patience: {ft_params.patience - steps_since_best}"
                )
                if steps_since_best >= ft_params.patience:
                    logger("Early stopping.")
                    break


def main():
    parser = argparse.ArgumentParser(description="Finetune NLLB model.")
    parser.add_argument(
        "--config", type=str, required=True, help="Directory to save finetuned model"
    )
    args = parser.parse_args()
    with open(args.config) as reader:
        config = json.load(reader)

    ft_params = read_finetuning_params(config)
    experiment_dir = create_experiment_dir(config, args.config)
    bitexts = create_bitexts(config)
    save_permutation_map(bitexts["cipher_map"], Path(experiment_dir) / "ciphers.json")

    static_model = prepare_model_for_finetuning(ft_params)
    model = prepare_model_for_finetuning(ft_params)
    train_mix = MixtureOfTextAndGoalEncodings(
        bitexts["train"], static_model.model.encoder
    )
    dev_mix = MixtureOfTextAndGoalEncodings(bitexts["dev"], static_model.model.encoder)
    finetune(
        model,
        train_mix,
        dev_mix,
        experiment_dir,
        ft_params,
    )
    evaluate_experiment(experiment_dir)


if __name__ == "__main__":
    main()
