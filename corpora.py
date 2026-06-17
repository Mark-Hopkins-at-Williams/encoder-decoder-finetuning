from align import extract_phrase_pairs
import random
import torch
from torch.utils.data import DataLoader, IterableDataset
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, Tuple, List, Optional, Iterator

CorpusId = Tuple[str, str]  # typedef


class Corpus(IterableDataset):
    def __init__(self, filename: str):
        self.filename = filename

    def __iter__(self) -> Iterator[str]:
        with open(self.filename, "r", encoding="utf-8") as f:
            for line in f:
                yield line.rstrip("\n")


class TokenizedCorpus:
    def __init__(self, corpus, tokenizer, lang_code):
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.lang_code = lang_code

    def __iter__(self):
        corpus_iter = iter(self.corpus)
        for line in corpus_iter:
            yield self._tokenize(line)

    def _tokenize(self, line):
        tokens = self.tokenizer(line, lang_code=self.lang_code)
        return tokens


class EncipheredCorpus:
    def __init__(self, tokenized_corpus, token_permutation):
        self.tokenized_corpus = tokenized_corpus
        self.token_permutation = token_permutation

    def __iter__(self):
        corpus_iter = iter(self.tokenized_corpus)
        for token_sequence in corpus_iter:
            yield self._encipher(token_sequence)

    def _encipher(self, tokens):
        return [self.token_permutation(tok) for tok in tokens]


class Bitext(IterableDataset):
    def __init__(self, corpus1, corpus2, lines=None):
        self.corpus1 = corpus1
        self.corpus2 = corpus2
        self.lines = lines

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        current_line = 0
        for line1, line2 in zip(self.corpus1, self.corpus2):
            if self.lines is not None and current_line >= self.lines[1]:
                break
            elif self.lines is None or current_line >= self.lines[0]:
                yield line1, line2
            current_line += 1


class CodeswitchedBitext:
    def __init__(
        self,
        tokenized_bitext,
        alignment,
        randomizer=lambda ls: random.randint(0, len(ls) - 1),
    ):
        self.tokenized_bitext = tokenized_bitext
        self.alignment = alignment
        self.randomizer = randomizer

    def __iter__(self):
        bitext_iter = iter(self.tokenized_bitext)
        for i, (toks1, toks2) in enumerate(bitext_iter):
            alignment = self.alignment[i]
            pairs = extract_phrase_pairs(toks1, toks2, alignment)
            indices = list(range(len(pairs)))
            index_chosen = False
            while len(indices) > 0 and not index_chosen:
                j = self.randomizer(indices)
                span1_start, span1_end, span2_start, span2_end = pairs[indices[j]]
                if (
                    span1_start == 0
                    or span2_start == 0
                    or span1_end == len(toks1) - 1
                    or span2_end == len(toks2) - 1
                ):  # don't include lang code or end of sentence marker
                    indices = indices[:j] + indices[j + 1 :]
                else:
                    index_chosen = True
            if index_chosen:
                revised_toks1 = []
                for j in range(len(toks1)):
                    if j == span1_start:
                        revised_toks1.extend(toks2[span2_start : span2_end + 1])
                    elif j < span1_start or j > span1_end:
                        revised_toks1.append(toks1[j])
                toks1 = revised_toks1
            yield (toks1, toks2)


class BitextIterableDataset(IterableDataset):
    def __init__(self, bitext_iterable):
        self.bitext_iterable = bitext_iterable

    def __iter__(self):
        yield from self.bitext_iterable


class BatchedBitext:
    def __init__(self, bitext, batch_size, src_pad_token=0, tgt_pad_token=0):
        self.bitext = bitext
        self.batch_size = batch_size
        self.src_pad_token = src_pad_token
        self.tgt_pad_token = tgt_pad_token

    def collate_fn(self, batch):
        src, tgt = zip(*batch)
        src = [torch.tensor(x) for x in src]
        tgt = [torch.tensor(x) for x in tgt]
        src_padded = pad_sequence(
            src, batch_first=True, padding_value=self.src_pad_token
        )
        tgt_padded = pad_sequence(
            tgt, batch_first=True, padding_value=self.tgt_pad_token
        )
        src_map = {
            "input_ids": src_padded,
            "attention_mask": (src_padded != self.src_pad_token).int(),
        }
        tgt_map = {
            "input_ids": tgt_padded,
            "attention_mask": (tgt_padded != self.tgt_pad_token).int(),
        }
        return src_map, tgt_map

    def __iter__(self):
        loader = DataLoader(
            BitextIterableDataset(self.bitext),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            drop_last=False,
        )
        for batch in loader:
            yield batch


class MixtureOfBitexts:
    def __init__(
        self,
        bitexts: Dict[Tuple[str, str], Bitext],
        metadata: Dict[Tuple[str, str], Dict[str, str]],
        sampling_probs: Optional[List[float]] = None,
        only_once_thru: bool = False,
    ):
        self.bitexts = bitexts
        self.metadata = metadata
        self.keys = list(bitexts)
        self.batch_iters = {}
        for key in self.keys:
            self.batch_iters[key] = iter(self.bitexts[key])

        total = sum(sampling_probs) if sampling_probs else len(bitexts)
        self.sampling_probs = [
            p / total for p in (sampling_probs or [1.0] * len(bitexts))
        ]
        self.only_once_thru = only_once_thru
        self.completed_bitexts = set()

    def restart(self):
        self.completed_bitexts = set()
        for key in self.keys:
            self.batch_iters[key] = iter(self.bitexts[key])

    def __iter__(self):
        still_looping = True
        while still_looping:
            still_choosing = True
            while still_choosing and len(self.completed_bitexts) < len(self.keys):
                lang_pair = random.choices(self.keys, weights=self.sampling_probs, k=1)[
                    0
                ]
                try:
                    lang1_sents, lang2_sents = next(self.batch_iters[lang_pair])
                    still_choosing = False
                except StopIteration:
                    if self.only_once_thru:
                        self.completed_bitexts.add(lang_pair)
                    else:
                        self.batch_iters[lang_pair] = iter(self.bitexts[lang_pair])
            if not still_choosing:
                yield lang1_sents, lang2_sents, self.metadata[lang_pair]
            else:
                still_looping = False

    def get_language_codes(self) -> List[str]:
        return sorted({code for pair in self.keys for code in pair})


class MixtureOfTextAndGoalEncodings:
    def __init__(self, mix, encoder):
        self.mix = mix
        self.encoder = encoder
        self.encoder.eval()

    def __iter__(self):
        for lang1_sents, lang2_sents, metadata in self.mix:
            lang2_sents = {k: v.to(self.encoder.device) for k, v in lang2_sents.items()}
            encodings = self.encoder(**lang2_sents).last_hidden_state
            yield lang1_sents, metadata["lang1_code"], encodings, lang2_sents[
                "attention_mask"
            ]

    def restart(self):
        self.mix.restart()
