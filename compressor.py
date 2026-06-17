import torch
import torch.nn.functional as F
import os
from autocomplete import initialize_model
from tokenization import Tokenizer
import sys


def get_predictions(
    logits, prediction_threshold, mode, temperature=1, id_for_incorrect=-1
):
    if mode == "top_pred":
        probs = F.softmax(logits / temperature, dim=-1)  # converting each logit to %s
        probs, preds = torch.max(
            probs, dim=-1
        )  # one matrix for the each char prediction and another matrix for "confidence" of each prediction
        preds = torch.where(
            probs >= prediction_threshold, preds, torch.tensor(id_for_incorrect)
        )  # (B,T)
        # Compare all of probs to all of threshold. If the prediction is sufficiently confident, the prediction is used for the element. Otherwise, id_for_unknown used
        return preds
    else:
        probs = F.softmax(logits / temperature, dim=-1)  # converting each logit to %s
        top_probs, top_preds = torch.topk(
            probs, k=2, dim=-1
        )  # # one matrix for top 2 char predictions and another matrix for "confidence" of these predictions

        top_guess_confidence = top_probs[:, :, 0]  # all probs of top guess
        second_guess_confidence = top_probs[:, :, 1]  # all probs of second guess
        top_guess = top_preds[:, :, 0]  # the actual top guesses

        margin = (
            top_guess_confidence - second_guess_confidence
        )  # gap between 1st and 2nd guesses
        preds = torch.where(
            margin >= prediction_threshold, top_guess, torch.tensor(id_for_incorrect)
        )
        # If the prediction is sufficiently confident, the prediction is used for the element. Otherwise, id_for_unknown used
        return preds


class AutocompletingTokenizer(Tokenizer):
    def __init__(self, model, device, max_length=None, max_length_encoding=1):
        self.model = model.to(device)
        self.device = device
        self.max_length_encoding = max_length_encoding
        self.special_tokens = {"<pad>": 0, "lang_Lang": 1, "</s>": 2}
        self.max_length = max_length
        self.prediction_threshold = 0.8
        self.prediction_mode = "top_pred"
        self.tokens_processed = 0
        self.tokens_outputted = 0
        self.sents_processed = 0
        self.omitted = dict()
        self.total = dict()

    def __call__(self, sent: str, lang_code="lang_Lang"):
        input_ids = [self.special_tokens[lang_code]] + [
            byte + len(self.special_tokens) for byte in sent.encode()
        ]
        self.sents_processed += 1

        if self.sents_processed % 1000000 == 0:
            print(
                f"COMPRESSION RATIO: {self.tokens_outputted / self.tokens_processed: .3f}"
            )
            sys.stdout.flush()

        self.tokens_processed += len(input_ids)
        target_ids = [byte + len(self.special_tokens) for byte in sent.encode()] + [
            self.special_tokens["</s>"]
        ]

        input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.device)
        target_ids = torch.tensor(target_ids).unsqueeze(0).to(self.device)
        logits = self.model(input_ids)  # (B, T, vocab_size)
        preds = get_predictions(
            logits,
            self.prediction_threshold,
            mode=self.prediction_mode,
            temperature=1,
            id_for_incorrect=-1,
        ).squeeze()
        input_ids = input_ids.squeeze()
        target_ids = target_ids.squeeze()
        target_ids = target_ids[:1024]  # TODO: make more robust
        correct = preds == target_ids
        tokens = []
        correct_accumulator = 0

        for j in range(len(correct) - 1, -1, -1):
            if correct[j]:
                correct_accumulator += 1
            else:
                offset = 256 * min(correct_accumulator, self.max_length_encoding)
                tokens.append(target_ids[j].item() + offset)
                correct_accumulator = 0
        tokens.append(input_ids[0].item())
        tokens.reverse()

        self.tokens_outputted += len(tokens)
        if self.max_length is not None and len(tokens) > self.max_length - 1:
            tokens = tokens[: self.max_length - 1] + [self.special_tokens["</s>"]]

        return tokens

    def __len__(self):
        return 256 + len(self.special_tokens) + (256 * self.max_length_encoding)

    def get_special_tokens(self):
        return self.special_tokens

    def batch_decode(self, token_ids):
        pass


def load_autocompleting_tokenizer(model_dir, max_length, max_length_encoding):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_model(max_length=1024)
    checkpoint = torch.load(
        os.path.join(model_dir, "best_model.pt"), map_location="cpu"
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return AutocompletingTokenizer(
        model, device, max_length=max_length, max_length_encoding=max_length_encoding
    )


if __name__ == "__main__":

    max_length = 192
    model_dir = "experiments/autocomplete-v1"

    tokenizer = load_autocompleting_tokenizer(model_dir, max_length)
    with open("test_files/lang1.txt") as reader:
        lines = [line.strip() for line in reader.readlines()]

    numerator, denominator = 0, 0
    for line in lines:
        tokens = tokenizer(line, lang_code="eng_Latn")
        numerator += len(tokens)
        denominator += len(line) + 2
        # for tok in tokens:
        #     if tok == 1:
        #         print("eng_Latn")
        #     elif tok == 2:
        #         print("</s>")
        #     elif 3 <= tok <= 3 + 256:
        #         print(chr(tok - 3) + "*")
        #     else:
        #         print(chr(tok - 259))
    print(numerator / denominator)
