import argparse
from configure import USE_CUDA
from corpora import MixtureOfBitexts, TokenizedMixtureOfBitexts, load_tokenizer
import json
import faiss
import torch
from transformers import AutoModelForSeq2SeqLM


def evaluate(model, dev_data, batches: int = 100):
    model.eval()
    encoder = model.model.encoder
    with torch.no_grad():
        for _ in range(batches):
            x, y, _, _ = dev_data.next_batch()
            x = x.to(model.device)
            y = y.to(model.device)
            x_encoding = encoder(**x)
            y_encoding = encoder(**y)
            xq = x_encoding.last_hidden_state[0]  # .cpu().numpy()
            xb = y_encoding.last_hidden_state[0]  # .cpu().numpy()
            print(xq.shape)
            print(xb.shape)
            index = faiss.IndexFlatL2(1024)
            index.add(xb)
            k = 1
            D, I = index.search(
                xq, k
            )  # D = distances, I = indices of nearest neighbors
            print(I)
            print(D)
            exit()


def main():
    parser = argparse.ArgumentParser(description="Finetune NLLB model.")
    parser.add_argument(
        "--config", type=str, required=True, help="Directory to save finetuned model"
    )
    args = parser.parse_args()

    with open(args.config) as reader:
        config = json.load(reader)

    lang_codes = dict()
    for corpus in config["corpora"]:
        for key in config["corpora"][corpus]:
            lang_codes[(corpus, key)] = config["corpora"][corpus][key]["lang_code"]
    model_name = "facebook/nllb-200-distilled-600M"

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if USE_CUDA:
        model.cuda()

    dev_data = MixtureOfBitexts.create_from_config(config, "dev", only_once_thru=True)
    tokenizer = load_tokenizer(model_name)

    tokenized_dev = TokenizedMixtureOfBitexts(
        dev_data,
        tokenizer,
        max_length=128,
        lang_codes=lang_codes,
        permutation_map=dict(),
    )

    evaluate(model, tokenized_dev, batches=2)


if __name__ == "__main__":
    main()
