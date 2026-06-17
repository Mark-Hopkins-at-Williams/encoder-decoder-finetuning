import argparse
import json
from pathlib import Path

import torch
import matplotlib
import matplotlib.pyplot as plt

from attention import SimpleAttention
from configure import USE_CUDA
from configure import harvest_language_codes
from configure import initialize_tokenizer
from configure import read_finetuning_params
from corpora import MixtureOfBitexts
from corpora import TokenizedMixtureOfBitexts
from corpora import TokenizedMixtureOfTextAndGoalEncoding
from myutil import prepare_model_for_finetuning, logger
from permutations import load_permutation_map
from transformers import AutoModelForSeq2SeqLM

matplotlib.use("Agg")


# -------------------------
# Visualization utilities
# -------------------------


def visualize_attention_grid_batch(
    weights: torch.Tensor,
    save_path: Path,
    title: str,
    ncols: int = 4,
):
    """
    Visualize a batch of attention matrices in a grid.

    weights: [B, T_q, T_k]
    Each subplot corresponds to one sentence's attention matrix.
    """
    weights = weights.detach().cpu().numpy()
    B = weights.shape[0]

    ncols = min(ncols, B)
    nrows = (B + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4 * ncols, 4 * nrows),
        squeeze=False,
    )

    vmin = weights.min()
    vmax = weights.max()

    im = None
    for i in range(nrows * ncols):
        ax = axes[i // ncols][i % ncols]
        if i < B:
            im = ax.imshow(
                weights[i],
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(f"sent {i}")
            ax.set_xlabel("Key tokens")
            ax.set_ylabel("Query tokens")
        else:
            ax.axis("off")

    fig.suptitle(title)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close()


def visualize_attention_batch(
    attn,
    enc_a,
    enc_b,
    mask_a,
    mask_b,
    out_dir: Path,
    prefix: str,
    batch_idx: int,
):
    """
    Visualize attention grids for all sentences in a batch
    as a single image.

    enc_a: [B, T_a, H]
    enc_b: [B, T_b, H]
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        _, weights = attn(enc_a, enc_b, mask_a, mask_b)
        # weights: [B, T_a, T_b]

    save_path = out_dir / f"{prefix}_batch{batch_idx}.png"

    visualize_attention_grid_batch(
        weights,
        save_path,
        title=f"{prefix} | batch {batch_idx}",
        ncols=4,
    )


# -------------------------
# Main driver
# -------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Visualize SimpleAttention alignments from config"
    )
    parser.add_argument("--exp_dir", type=str, required=True)
    args = parser.parse_args()

    experiment_dir = args.exp_dir
    logger(f"Initializing model from: {experiment_dir}")

    config_file = Path(experiment_dir) / "experiment.json"
    with open(config_file) as reader:
        config = json.load(reader)

    ft_params = read_finetuning_params(config)

    model = AutoModelForSeq2SeqLM.from_pretrained(experiment_dir)
    if USE_CUDA:
        model.cuda()

    static_model = prepare_model_for_finetuning(ft_params)

    lang_codes = harvest_language_codes(config)
    tokenizer = initialize_tokenizer(config)
    pmap = load_permutation_map(Path(experiment_dir) / "permutations.json")

    dev_data = MixtureOfBitexts.create_from_config(config, "dev", only_once_thru=True)

    tokenized_dev = TokenizedMixtureOfBitexts(
        dev_data,
        tokenizer,
        lang_codes=lang_codes,
        permutation_map=pmap,
        use_alt_pad_token_for_tgt_lang=False,
    )

    dev_mix = TokenizedMixtureOfTextAndGoalEncoding(
        tokenized_dev, static_model.model.encoder
    )

    encoder = model.model.encoder
    encoder.eval()

    viz_root = Path(experiment_dir) / "attention_visualizations"
    viz_root.mkdir(parents=True, exist_ok=True)

    logger("Starting attention visualization over dev set")

    attn = SimpleAttention()
    batch_idx = 0

    with torch.no_grad():
        batch = dev_mix.next_batch()
        while batch is not None:
            sents, lang, goal_encodings, goal_attn_mask = batch

            sents = sents.to(encoder.device)
            goal_encodings = goal_encodings.to(encoder.device)

            sent_encodings = encoder(**sents).last_hidden_state

            lang_dir = viz_root / f"{lang[0]}_{lang[1]}"
            lang_dir.mkdir(parents=True, exist_ok=True)

            visualize_attention_batch(
                attn,
                sent_encodings,
                goal_encodings,
                sents["attention_mask"],
                goal_attn_mask,
                out_dir=lang_dir,
                prefix="sent_to_goal",
                batch_idx=batch_idx,
            )

            visualize_attention_batch(
                attn,
                goal_encodings,
                sent_encodings,
                goal_attn_mask,
                sents["attention_mask"],
                out_dir=lang_dir,
                prefix="goal_to_sent",
                batch_idx=batch_idx,
            )

            batch_idx += 1
            batch = dev_mix.next_batch()

            if batch_idx > 5:
                break

    logger("Attention visualization complete")


if __name__ == "__main__":
    main()
