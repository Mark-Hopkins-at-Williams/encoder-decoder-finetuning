import argparse
from attention import SimpleAttention
from configure import USE_CUDA
from configure import harvest_language_codes
from configure import initialize_tokenizer
from configure import read_finetuning_params
from corpora import MixtureOfBitexts
from corpora import TokenizedMixtureOfBitexts
from corpora import TokenizedMixtureOfTextAndGoalEncoding
import json
import matplotlib
import matplotlib.pyplot as plt
from myutil import prepare_model_for_finetuning, logger
from pathlib import Path
from permutations import load_permutation_map
import torch
from transformers import AutoModelForSeq2SeqLM
import torch.nn.functional as F

matplotlib.use("Agg")


# -------------------------
# Visualization utilities
# -------------------------


def visualize_attention_grid(
    weights: torch.Tensor,
    save_path: Path,
    title: str,
):
    """
    Visualize a single attention matrix.

    weights: [T_q, T_k]
    """
    weights = weights.detach().cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(weights, aspect="auto")
    plt.colorbar(label="attention weight")

    plt.xlabel("Key tokens (other encoding)")
    plt.ylabel("Query tokens (current encoding)")
    plt.title(title)

    plt.tight_layout()
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
    Visualize attention grids for every sentence in a batch.

    enc_a: [B, T_a, H]
    enc_b: [B, T_b, H]
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        _, weights = attn(enc_a, enc_b, mask_a, mask_b)
        # weights: [B, T_a, T_b]

    B = weights.size(0)

    for i in range(B):
        save_path = out_dir / f"{prefix}_batch{batch_idx}_sent{i}.png"
        visualize_attention_grid(
            weights[i],
            save_path,
            title=f"{prefix} | batch {batch_idx} | sent {i}",
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
            print("goal:")
            print(goal_encodings[0][6])

            scaled_goal = goal_encodings * (0.0728 / 0.4713)
            print("scaled goal:")
            print(scaled_goal[0][6])

            sent_encodings = encoder(**sents).last_hidden_state
            print("sent:")
            print(sent_encodings[0][6])
            print(
                F.cosine_similarity(sent_encodings[0][0], goal_encodings[0][0], dim=0)
            )

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
