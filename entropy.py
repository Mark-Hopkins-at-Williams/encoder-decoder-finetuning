import argparse
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from corpora import MixtureOfBitexts, TokenizedMixtureOfBitexts
import torch

def extract_smoothed_unigram_lm(dataset, vocab_size, alpha=1.0, ignore_token_ids=None):
    """
    Builds a smoothed unigram language model from a tokenized dataset.
    
    Args:
        dataset: iterable of examples with 'input_ids' key
        vocab_size: tokenizer.vocab_size or len(tokenizer)
        alpha: Laplace smoothing constant (typically 1.0)
        ignore_token_ids: optional set of token IDs to ignore (e.g., special tokens)

    Returns:
        probs: torch.Tensor of shape (vocab_size,) with smoothed probabilities
    """
    counts = torch.zeros(vocab_size, dtype=torch.float64)
    total_tokens = 0

    batch = dataset.next_batch()
    while batch is not None:
        _, y, _, _ = batch
        input_ids = y["input_ids"]
        for token_id in input_ids:
            if ignore_token_ids and token_id in ignore_token_ids:
                continue
            counts[token_id] += 1
            total_tokens += 1
        batch = dataset.next_batch()

    # Apply Laplace smoothing
    smoothed_counts = counts + alpha
    smoothed_total = total_tokens + alpha * vocab_size
    probs = smoothed_counts / smoothed_total

    return probs


def evaluate(model, dev_data):
    model.eval()
    with torch.no_grad():
        batch = dev_data.next_batch()
        while batch is not None:
            x, y, _, _ = batch
            x = x.to(model.device)
            y = y.to(model.device)

            output = model(**x, labels=y.input_ids)
            logits = output.logits  # (batch_size, tgt_seq_len, vocab_size)
            labels = y.input_ids    # (batch_size, tgt_seq_len)

            safe_labels = labels.clone()
            safe_labels[safe_labels == -100] = 0  # dummy token for gather

            log_probs = F.log_softmax(logits, dim=-1)
            logp_correct = log_probs.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)

            non_pad_mask = labels != -100
            nll = -logp_correct
            masked_nll = nll * non_pad_mask

            total_loss = masked_nll.sum() / non_pad_mask.sum()

            print("Manual loss:", total_loss.item())
            print("Hugging Face loss:", output.loss.item())

            batch = dev_data.next_batch()
            batch = None  # remove this in real use

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate translation model using entropy metrics.")
    parser.add_argument(
        "--config", type=str, required=True, help="Configuration JSON file."
    )
    args = parser.parse_args()
    with open(args.config) as reader:
        config = json.load(reader)
    
    lang_codes = dict()        
    for corpus in config['corpora']:
        for key in config['corpora'][corpus]:
            lang_codes[(corpus, key)] = config['corpora'][corpus][key]['lang_code']
    
    model_name = config['finetuning_parameters']['base_model']
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dev_data = MixtureOfBitexts.create_from_config(config, "dev", only_once_thru=True)
    tokenized_dev = TokenizedMixtureOfBitexts(
        dev_data, tokenizer, max_length=128, lang_codes=lang_codes
    )
    probs = extract_smoothed_unigram_lm(tokenized_dev, len(tokenizer), ignore_token_ids=tokenizer.all_special_tokens)
    print(probs.shape)
    print(probs)
    #print(evaluate(model, tokenized_dev))